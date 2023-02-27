# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Geometry utilities for working with LayoutLM, Amazon Textract, and SageMaker Ground Truth
"""
# Python Built-Ins:
from typing import Iterable, Optional, Union

# External Dependencies:
import numpy as np
import torch
import trp


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
    return_tensors: Optional[str] = None,
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
    return_tensors :
        None (default) to return plain nested lists of ints. "np" to return a numpy array or "pt"
        to return a torch.LongTensor.

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
        raise ValueError(
            "return_tensors must be 'np' or 'pt' for layoutlm_boxes_from_trp_blocks(). Got: %s"
            % return_tensors
        )
