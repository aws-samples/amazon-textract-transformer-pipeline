# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Geometry utilities for working with LayoutLM, Amazon Textract, and SageMaker Ground Truth
"""
# Python Built-Ins:
from typing import Iterable, Union

# External Dependencies:
import numpy as np
import torch
import trp


class AnnotationBoundingBox:
    """Class to parse a bounding box annotated by SageMaker Ground Truth Object Detection

    Pre-calculates all box TLHWBR metrics (both absolute and relative) on init, for efficient and
    easy processing later.
    """

    def __init__(self, manifest_box: dict, image_height: int, image_width: int):
        self._class_id = manifest_box["class_id"]
        self._abs_top = manifest_box["top"]
        self._abs_left = manifest_box["left"]
        self._abs_height = manifest_box["height"]
        self._abs_width = manifest_box["width"]
        self._abs_bottom = self.abs_top + self.abs_height
        self._abs_right = self.abs_left + self.abs_width
        self._rel_top = self._abs_top / image_height
        self._rel_left = self._abs_left / image_width
        self._rel_height = self._abs_height / image_height
        self._rel_width = self._abs_width / image_width
        self._rel_bottom = self._abs_bottom / image_height
        self._rel_right = self._abs_right / image_width

    @property
    def class_id(self):
        return self._class_id

    @property
    def abs_top(self):
        return self._abs_top

    @property
    def abs_left(self):
        return self._abs_left

    @property
    def abs_height(self):
        return self._abs_height

    @property
    def abs_width(self):
        return self._abs_width

    @property
    def abs_bottom(self):
        return self._abs_bottom

    @property
    def abs_right(self):
        return self._abs_right

    @property
    def rel_top(self):
        return self._rel_top

    @property
    def rel_left(self):
        return self._rel_left

    @property
    def rel_height(self):
        return self._rel_height

    @property
    def rel_width(self):
        return self._rel_width

    @property
    def rel_bottom(self):
        return self._rel_bottom

    @property
    def rel_right(self):
        return self._rel_right


class BoundingBoxAnnotationResult:
    """Class to parse the result field saved by a SageMaker Ground Truth Object Detection job"""

    def __init__(self, manifest_obj: dict):
        """Initialize a BoundingBoxAnnotationResult

        Arguments
        ---------
        manifest_obj : dict
            The contents of the output field of a record in a SMGT Object Detection labelling job
            output manifest, or equivalent.
        """
        try:
            image_size_spec = manifest_obj["image_size"][0]
            self._image_height = int(image_size_spec["height"])
            self._image_width = int(image_size_spec["width"])
            self._image_depth = (
                int(image_size_spec["depth"]) if "depth" in image_size_spec else None
            )
        except Exception as e:
            raise ValueError(
                "".join(
                    (
                        "manifest_obj must be a dictionary including 'image_size': a list of ",
                        "length 1 whose first/only element is a dict with integer properties ",
                        f"'height' and 'width', optionally also 'depth'. Got: {manifest_obj}",
                    )
                )
            ) from e
        assert (
            len(manifest_obj["image_size"]) == 1
        ), f"manifest_obj['image_size'] must be a list of len 1. Got: {manifest_obj['image_size']}"

        try:
            self._boxes = [
                AnnotationBoundingBox(
                    b,
                    image_height=self._image_height,
                    image_width=self._image_width,
                )
                for b in manifest_obj["annotations"]
            ]
        except Exception as e:
            raise ValueError(
                "".join(
                    (
                        "manifest_obj['annotations'] must be a list-like of absolute TLHW bounding box ",
                        f"dicts with class_id. Got {manifest_obj['annotations']}",
                    )
                )
            ) from e

    @property
    def image_height(self):
        return self._image_height

    @property
    def image_width(self):
        return self._image_width

    @property
    def image_depth(self):
        return self._image_depth

    @property
    def boxes(self):
        return self._boxes


def layoutlm_boxes_from_trp_blocks(
    textract_blocks: Iterable[
        Union[
            trp.Word,
            trp.Line,
            trp.SelectionElement,
            trp.FieldKey,
            trp.FieldValue,
            trp.Cell,
            trp.Table,
            trp.Page,
        ]
    ],
    return_tensors: str = "np",
):
    """List of TRP 'blocks' to array of 0-1000 normalized x0,y0,x1,y1 for LayoutLM

    Per https://docs.aws.amazon.com/textract/latest/dg/API_BoundingBox.html, Textract bounding box
    coords are 0-1 relative to page size already: So we just need to multiply by 1000. Note this
    means there's no information encoded about the overall aspect ratio of the page.

    Parameters
    ----------
    textract_blocks :
        Iterable of any Textract TRP objects including a 'geometry' property e.g. Word, Line, Cell,
        etc.
    return_tensors : str
        Either "np" (default) to return a numpy array or "pt" to return a torch.LongTensor.

    Returns
    -------
    boxes :
        Array or tensor shape (n_examples, 4) of bounding boxes: left, top, right, bottom scaled
        0-1000.
    """
    raw_zero_to_one_list = [
        [
            block.geometry.boundingBox.left,
            block.geometry.boundingBox.top,
            block.geometry.boundingBox.left + block.geometry.boundingBox.width,
            block.geometry.boundingBox.top + block.geometry.boundingBox.height,
        ]
        for block in textract_blocks
    ]
    if return_tensors == "np":
        return np.array(raw_zero_to_one_list) * 1000
    elif return_tensors == "pt":
        return (torch.FloatTensor(raw_zero_to_one_list) * 1000).long()
    else:
        raise ValueError(
            "return_tensors must be 'np' or 'pt' for layoutlm_boxes_from_trp_blocks(). Got: %s"
            % return_tensors
        )
