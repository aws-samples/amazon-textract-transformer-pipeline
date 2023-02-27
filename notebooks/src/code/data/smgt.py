# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Data models for working with SageMaker Ground Truth in general and our specific custom task UI.

Includes parsing e.g. bounding box results from built-in task type or the crowd-bounding-box tag.
"""
# Python Built-Ins:
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

# External Dependencies:
import numpy as np
import torch


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

    def normalized_boxes(
        self,
        return_tensors: Optional[str] = None,
    ):
        """Annotation boxes in 0-1000 normalized x0,y0,x1,y1 array/tensor format as per LayoutLM"""
        raw_zero_to_one_list = [
            [
                box.rel_left,
                box.rel_top,
                box.rel_right,
                box.rel_bottom,
            ]
            for box in self._boxes
        ]
        if return_tensors == "np" or not return_tensors:
            if len(raw_zero_to_one_list) == 0:
                npresult = np.zeros((0, 4), dtype="long")
            else:
                npresult = (np.array(raw_zero_to_one_list) * 1000).astype("long")
            return npresult if return_tensors else npresult.tolist()
        elif return_tensors == "pt":
            if len(raw_zero_to_one_list) == 0:
                return torch.zeros((0, 4), dtype=torch.long)
            else:
                return (torch.FloatTensor(raw_zero_to_one_list) * 1000).long()
        else:
            raise ValueError("return_tensors must be None, 'np' or 'pt'. Got: %s" % return_tensors)


class OCRReviewStatus(str, Enum):
    """Ternary status for OCR transcription review

    TODO: Merge/share with postproc Lambda function if possible?
    """

    correct = "correct"
    unclear = "unclear"
    wrong = "wrong"


@dataclass
class OCREntityWithTranscriptReview:
    detection_id: str
    ocr_status: OCRReviewStatus
    box_ixs: List[int]
    class_id: int  # TODO: This is optional in postproc Lambda's data model, re-align
    raw_text: str  # TODO: This is optional in postproc Lambda's data model, re-align
    target_text: Optional[str]
    label: Optional[str]

    @classmethod
    def from_dict(cls, raw: dict) -> OCREntityWithTranscriptReview:
        """Parse an individual entity annotation as produced by custom SMGT task UI+post-proc"""
        raw_text = raw["rawText"]
        ocr_status = OCRReviewStatus[raw["ocrStatus"]]
        target_text = raw.get("targetText")
        if target_text is None:
            if ocr_status != OCRReviewStatus.wrong:
                target_text = raw_text
            else:
                raise ValueError(
                    "Entity annotation is missing targetText field, but is tagged with ocrStatus "
                    "'wrong' so we can't take the rawText as target: %s" % raw
                )

        return cls(
            detection_id=raw["detectionId"],
            ocr_status=ocr_status,
            box_ixs=raw["boxIxs"],
            class_id=raw["classId"],
            raw_text=raw_text,
            target_text=target_text,
            label=raw.get("label"),
        )


class BBoxesWithTranscriptReviewsAnnotationResult(BoundingBoxAnnotationResult):
    """Result field saved by an SMGT job using the custom entities-with-transcription-reviews task

    This custom task, introduced via the notebooks and implemented by custom Liquid HTML template
    and pre/post-processing Lambda functions, outputs data compatible with the standard bounding box
    task UI but enriched with consolidated (overlapping) per-class regions and transcription reviews
    for each region.
    """

    entities: List[OCREntityWithTranscriptReview]

    def __init__(self, manifest_obj: dict):
        # Parse the bounding boxes themselves via superclass:
        super().__init__(manifest_obj)
        # Parse the OCR entities:
        if "entities" not in manifest_obj:
            raise ValueError(
                "SMGT manifest is missing 'entities' key, which should be generated by the custom "
                "task template but not the built-in bounding box UI. Using bbox-only annotations "
                "for a seq2seq training job is not currently supported."
            )
        self.entities = [
            OCREntityWithTranscriptReview.from_dict(e) for e in manifest_obj["entities"]
        ]
