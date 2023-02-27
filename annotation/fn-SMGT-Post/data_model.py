# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Parsers, data models, and utilities for our custom (OCR-oriented) components

This module contains code for parsing and translating data objects from our custom OCR review
SMGT task template, ready for consolidation into the final output manifest.
"""
# Python Built-Ins:
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import json
from logging import getLogger
import re
from typing import List, Optional

# Local Dependencies:
from smgt import BaseJsonable, BaseObjectParser, SMGTOutputBoundingBox

logger = getLogger("data_model")


class OCRReviewStatus(str, Enum):
    """Ternary status for OCR transcription review"""

    correct = "correct"
    unclear = "unclear"
    wrong = "wrong"


@dataclass
class SMGTOCREntity(BaseJsonable, BaseObjectParser):
    """BBox+transcript review OCR entity annotation, as used in consolidation

    This class `parse()`s from raw template output format and serializes to final output manifest
    format - so it's a bit specific to consolidation/post-processing Lambda as written.

    Attributes
    ----------
    detection_id :
        Auto-generated identifier assigned to each bounding box cluster/group by the UI template.
    ocr_status :
        Parsed status of the OCR transcription review (correct, unclear, wrong).
    box_ixs :
        Indexes of the bounding boxes in the main crowd-bounding-box result that this entity
        corresponds to.
    class_id :
        Numeric ID of the entity type/class (either this or string label should be known).
    label :
        String name of the entity type/class (either this or the numeric class_id should be known).
    raw_text :
        The raw text for the entity as detected by OCR tool.
    target_text :
        The target/normalized text as overridden by the user.
    """

    detection_id: str
    ocr_status: OCRReviewStatus
    box_ixs: List[int]
    class_id: Optional[int] = None
    label: Optional[str] = None
    raw_text: Optional[str] = None
    target_text: Optional[str] = None

    @classmethod
    def find_detection_ids(cls, parent_obj: dict) -> List[str]:
        """Find all auto-generated entity/detection IDs in top-level custom task output data

        Because of the mechanics of the SM Crowd HTML Elements and the template, there are multiple
        keys in the annotation output storing each entity's raw data. This function discovers
        available entity/detection IDs in a result.

        Parameters
        ----------
        parent_obj :
            Top-level annotation data object as output by the UI task template, containing multiple
            fields.
        """
        return sorted(
            set(
                map(
                    lambda m: m.group(1),
                    filter(
                        lambda m: m,
                        map(
                            lambda key: re.match(r"ocr-(.*)-[a-z]+", key, flags=re.IGNORECASE),
                            parent_obj.keys(),
                        ),
                    ),
                ),
            ),
        )

    @classmethod
    def parse(
        cls,
        parent_obj: dict,
        detection_id: str,
        boxes: Optional[List[SMGTOutputBoundingBox]] = None,
    ) -> SMGTOCREntity:
        """Parse the entity with given ID from the *whole annotation object*

        Use the `find_detection_ids()` method to look up available IDs in the top-level annotation
        data, then this parser to extract each ID.

        Parameters
        ----------
        parent_obj :
            Top-level annotation data object as output by the UI task template, containing multiple
            fields.
        detection_id :
            Specific entity/group ID to extract for this entity
        boxes :
            If provided, these will simply be used to validate the tagged `boxIxs` in the entity
            annotation are within range of the crowd-bounding-box tool's output.

        Raises
        ------
        ValueError
            If missing data or inconsistencies prevent the entity from being parsed from raw data.
        """
        meta_field_key = f"ocr-{detection_id}-meta"
        if meta_field_key not in parent_obj:
            raise ValueError(
                "OCR annotation metadata key %s not found in raw data" % meta_field_key,
            )

        meta = json.loads(parent_obj[meta_field_key])
        box_ixs = meta["boxIxs"]
        if len(box_ixs) < 1:
            raise ValueError(
                "OCR annotation has no linked box annotations: %s" % detection_id,
            )
        label = meta.get("label")
        class_id = meta.get("labelId")
        raw_text = meta.get("ocrText")
        if boxes is not None:
            n_boxes = len(boxes)
            illegal_box_ixs = [ix >= 0 and ix < n_boxes for ix in box_ixs]
            if len(illegal_box_ixs) > 0:
                raise ValueError(
                    "OCR annotation '%s' links to boxIxs outside the range 0-%s: %s"
                    % (detection_id, n_boxes, illegal_box_ixs)
                )
            if label is None:
                label = boxes[box_ixs[0]].label
            if class_id is None:
                class_id = boxes[box_ixs[0]].class_id

        OCR_STATUSES = tuple(s.value for s in OCRReviewStatus)  # String enum to Tuple[str]
        ocr_status_fields = [f"ocr-{detection_id}-{status}" for status in OCR_STATUSES]
        unknown_statuses = [
            s for ix, s in enumerate(OCR_STATUSES) if ocr_status_fields[ix] not in parent_obj
        ]
        if len(unknown_statuses):
            logger.warning(
                "OCR annotation %s could not determine whether the following statuses were "
                "selected: %s",
                detection_id,
                unknown_statuses,
            )
        selected_statuses = [
            s
            for ix, s in enumerate(OCR_STATUSES)
            if parent_obj.get(ocr_status_fields[ix], {}).get("on")
        ]
        n_selected_statuses = len(selected_statuses)
        if n_selected_statuses == 1:
            parsed_status = OCRReviewStatus[selected_statuses[0]]
        elif n_selected_statuses >= 1:
            logger.warning(
                "OCR annotation %s selected %s statuses: %s. Marking as 'unclear'",
                detection_id,
                n_selected_statuses,
                selected_statuses,
            )
            parsed_status = OCRReviewStatus.unclear
        else:  # (0 selected statuses)
            logger.warning(  # TODO: push warnings through to output manifest?
                "Missing OCR review status for annotation %s. Assuming 'unclear'",
                detection_id,
            )
            parsed_status = OCRReviewStatus.unclear

        if parsed_status == OCRReviewStatus.correct:
            target_text = raw_text
        else:
            correction_field_key = f"ocr-{detection_id}-override"
            target_text = parent_obj.get(correction_field_key)
            if parsed_status == OCRReviewStatus.wrong and correction_field_key not in parent_obj:
                logger.warning(
                    "OCR annotation %s tagged as 'wrong', but target text field %s is missing",
                    detection_id,
                    correction_field_key,
                )

        return SMGTOCREntity(
            detection_id=detection_id,
            ocr_status=parsed_status,
            box_ixs=box_ixs,
            class_id=class_id,
            label=label,
            raw_text=raw_text,
            target_text=target_text,
        )

    def to_jsonable(self) -> dict:
        return {
            k: v
            for k, v in {
                "detectionId": self.detection_id,
                "ocrStatus": self.ocr_status,
                "boxIxs": self.box_ixs,
                "classId": self.class_id,
                "label": self.label,
                "rawText": self.raw_text,
                "targetText": self.target_text,
            }.items()
            if v is not None
        }


@dataclass
class SMGTWorkerAnnotation(BaseJsonable, BaseObjectParser):
    """One worker's full annotation for a page using the custom bbox+transcript review task UI

    This class `parse()`s from raw template output format and serializes to final output manifest
    format - so it's a bit specific to consolidation/post-processing Lambda as written.

    Attributes
    ----------
    boxes :
        Parsed SMGT crowd-bounding-box boxes as labelled
    entities :
        Parsed OCR "entities" (bounding box groupings with transcription accuracy reviews)
    image_height :
        Input image height in pixels
    image_width :
        Input image width in pixels
    image_depth :
        Input image number of channels (usually 1 grayscale or 3 RGB) if known.
    """

    boxes: List[SMGTOutputBoundingBox]
    entities: List[SMGTOCREntity]
    image_height: int
    image_width: int
    image_depth: Optional[int] = None

    @classmethod
    def parse(
        cls,
        obj: dict,
        class_list: Optional[List[str]] = None,
        crowd_bounding_box_name: str = "boxtool",
    ) -> SMGTWorkerAnnotation:
        boxtool_data = obj[crowd_bounding_box_name]
        image_props = boxtool_data["inputImageProperties"]
        image_height = image_props["height"]
        image_width = image_props["width"]
        image_depth = image_props.get("depth")

        boxes = [
            SMGTOutputBoundingBox.parse(box, class_list=class_list)
            for box in boxtool_data["boundingBoxes"]
        ]
        entity_detection_ids = SMGTOCREntity.find_detection_ids(obj)
        entities = []
        for det_id in entity_detection_ids:
            try:
                entities.append(SMGTOCREntity.parse(obj, det_id))
            except Exception:
                logger.exception("Failed to load annotated entity %s", det_id)
                # TODO: Propagate failed entity extractions as warnings to output too?

        return cls(
            boxes=boxes,
            entities=entities,
            image_height=image_height,
            image_width=image_width,
            image_depth=image_depth,
        )

    def to_jsonable(self) -> dict:
        img_meta = {"height": self.image_height, "width": self.image_width}
        if self.image_depth is not None:
            img_meta["depth"] = self.image_depth
        return {
            # Image metadata and bounding boxes in format compatible with built-in BBox task:
            "image_size": [img_meta],
            "annotations": [box.to_jsonable() for box in self.boxes],
            # Additional data for OCR transcription reviews:
            "entities": [entity.to_jsonable() for entity in self.entities],
        }
