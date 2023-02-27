# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Parsers, data models, and utilities for generic SageMaker Ground Truth objects

This module contains code for dealing with SMGT intermediate formats and consolidation requests
in general (not specific to our particular custom template design).
"""
# Python Built-Ins:
from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
import json
import logging
from typing import List, Optional, Union
from urllib.parse import urlparse

# External Dependencies:
import boto3  # AWS SDK for Python

logger = logging.getLogger("smgt")
s3client = boto3.client("s3")


class BaseObjectParser(ABC):
    """Base interface for classes that can be created by parse()ing some (JSON?) object"""

    @classmethod
    def parse(cls, obj: Union[dict, float, int, list, str]) -> BaseObjectParser:
        raise NotImplementedError("Parsers must implement parse() method")


class BaseJsonable(ABC):
    """Base interface for classes that can be represented by a JSON-serializable object"""

    def to_jsonable(self) -> Union[dict, float, int, list, str]:
        raise NotImplementedError("BaseJsonable classes must implement to_jsonable() method")


class S3OrInlineObject:
    """Wrapper class for API dicts that contain either `content` (inline) or `s3Uri`"""

    def __init__(self, obj: dict):
        if "content" in obj:
            self._inline = True
            self._raw = obj["content"]
        elif "s3Uri" in obj:
            self._inline = False
            self._raw = obj["s3Uri"]
        else:
            raise ValueError("API object expected to contain either 'content' or 's3 key: %s" % obj)

    def fetch(self) -> Union[bytes, str]:
        """Load the text content (either inline or from S3 object)"""
        if self._inline:
            return self._raw
        else:
            logger.info("Fetching S3 object %s", self._raw)
            parsed_url = urlparse(self._raw)
            text_file = s3client.get_object(Bucket=parsed_url.netloc, Key=parsed_url.path[1:])
            return text_file["Body"].read()


@dataclass
class WorkerAnnotation(BaseObjectParser):
    """One worker's raw annotation for an object

    Attributes
    ----------
    worker_id :
        Opaque worker identifier, for example something like "private.us-east-1.e47e1e0123456789"
        for an internal workforce in us-east-1.
    """

    worker_id: str
    _annotation_data: S3OrInlineObject

    @classmethod
    def parse(cls, obj: dict) -> WorkerAnnotation:
        return cls(
            worker_id=obj["workerId"], _annotation_data=S3OrInlineObject(obj["annotationData"])
        )

    def fetch_data(self) -> dict:
        """Fetch (and JSON-parse) the worker's annotation for this object"""
        return json.loads(self._annotation_data.fetch())


@dataclass
class ObjectAnnotationResult(BaseObjectParser):
    """One dataset object's pre-consolidation annotations (from multiple workers)

    Attributes
    ----------
    dataset_object_id :
        Index of the object in the SMGT job dataset
    data_object :
        Main input object (i.e. just the source-ref or source, not the whole manifest line) for the
        task
    annotations :
        List of raw annotations from possibly multiple workers.
    """

    dataset_object_id: str
    data_object: S3OrInlineObject
    annotations: List[WorkerAnnotation]

    @classmethod
    def parse(cls, obj: dict) -> ObjectAnnotationResult:
        return cls(
            dataset_object_id=obj["datasetObjectId"],
            data_object=S3OrInlineObject(obj["dataObject"]),
            annotations=[WorkerAnnotation.parse(o) for o in obj["annotations"]],
        )


@dataclass
class ConsolidationRequest(BaseObjectParser):
    """Loaded `event` for this post-annotation Lambda function

    See:
    https://docs.aws.amazon.com/sagemaker/latest/dg/sms-custom-templates-step3-lambda-requirements.html#sms-custom-templates-step3-postlambda

    Attributes
    ----------
    version :
        A version number used internally by Ground Truth
    labelingJobArn :
        The Amazon Resource Name, or ARN, of your labeling job. This ARN can be used to reference
        the labeling job when using Ground Truth API operations such as DescribeLabelingJob.
    labelCategories :
        Includes the label categories and other attributes you either specified in the console, or
        that you include in the label category configuration file.
    labelAttributeName :
        Either the name of your labeling job, or the label attribute name you specify when you
        create the labeling job.
    roleArn :
        The Amazon Resource Name (ARN) of the IAM execution role you specify when you create the
        labeling job.
    payload :
        Raw annotation data for this request.
    """

    version: str
    labeling_job_arn: str
    label_categories: List[str]
    label_attribute_name: str
    role_arn: str
    payload: S3OrInlineObject

    @classmethod
    def parse(cls, obj: dict) -> ConsolidationRequest:
        return cls(
            version=obj["version"],
            labeling_job_arn=obj["labelingJobArn"],
            label_categories=obj.get("labelCategories", []),
            label_attribute_name=obj["labelAttributeName"],
            role_arn=obj["roleArn"],
            payload=S3OrInlineObject(obj["payload"]),
        )

    def fetch_object_annotations(self) -> List[ObjectAnnotationResult]:
        """Fetch and parse the list of object raw annotation data included for this request"""
        logger.info("Fetching consolidation request payload (raw annotation data)")
        payload_data = self.payload.fetch()
        logger.info("Parsing raw annotation list")
        payload_data = json.loads(payload_data)
        if not isinstance(payload_data, list):
            raise ValueError(
                "Expected consolidation request.payload to point to a JSON list file, but top-level"
                "object after parsing was of type: %s" % type(payload_data)
            )
        return [ObjectAnnotationResult.parse(object_data) for object_data in payload_data]


@dataclass
class PostConsolidationDatum(BaseJsonable):
    """Expected output format for each dataset object in the consolidation request"""

    dataset_object_id: str
    consolidated_content: Union[dict, BaseJsonable]

    def to_jsonable(self) -> dict:
        if hasattr(self.consolidated_content, "to_jsonable"):
            content = self.consolidated_content.to_jsonable()
        else:
            content = self.consolidated_content
        return {
            "datasetObjectId": self.dataset_object_id,
            "consolidatedAnnotation": {"content": content},
        }


@dataclass
class SMGTOutputBoundingBox(BaseJsonable, BaseObjectParser):
    """Maybe-slightly-customized SageMaker Ground Truth bounding box data model

    TODO: Review whether we should have two separate models here?

    SMGT built-in bounding box jobs produce boxes with numeric `class_id` in the output... But the
    crowd-bounding-box annotator tool produces raw outputs with string `label` to identify the
    selected class.

    It's possible to convert between the two in the post-annotation Lambda function,
    assuming your SMGT job was set up with the `LabelCategoryConfigS3Uri` parameter (in which case
    the Lambda will receive the class name list).

    *However*, the built-in task outputs a `"class-map": {"0": "Name0", ...}` key in the `-meta`
    field to link from numeric IDs back to string labels. AFAICT we can't output `-meta` field data
    with custom task post-processing Lambdas, as it seems to get overwritten in the output.

    So instead, this class allows you to specify the class list at parse() time and will serialize
    *both* class_id number and label string into the output box, if both are known.

    Attributes
    ----------
    top :
        Absolute top of the bounding box relative to the page origin (in pixels)
    left :
        Absolute left of the bounding box relative to the page origin (in pixels)
    height :
        Absolute height of the bounding box in pixels
    width :
        Absolute width of the bounding box in pixels
    label :
        String class/type name of the bounding box (if known)
    class_id :
        0-based integer class/type ID of the bounding box (if known)
    """

    top: int
    left: int
    height: int
    width: int
    label: Optional[str] = None
    class_id: Optional[int] = None

    def to_jsonable(self) -> dict:
        """Serialize the box to a JSON-able plain dictionary

        Whichever of `class_id` or `label` (or both) are known will be included.
        """
        result = {"top": self.top, "left": self.left, "height": self.height, "width": self.width}
        if self.class_id is not None:
            result["class_id"] = self.class_id
        if self.label is not None:
            result["label"] = self.label
        return result

    @classmethod
    def parse(cls, obj: dict, class_list: Optional[List[str]] = None) -> SMGTOutputBoundingBox:
        """Parse the bounding box from a SMGT box dictionary

        Parameters
        ----------
        obj :
            Dictionary specifying the box as generated by SageMaker Ground Truth labelling tool
        class_list :
            Optional list of class names, which if provided will be used to map between numeric
            `class_id` and string `label` in cases where only one is provided in the raw data.
        """
        label = obj.get("label")
        class_id = obj.get("class_id")
        if class_list and len(class_list) > 0:
            if label is None and class_id is not None:
                if class_id >= 0 and class_id < len(class_list):
                    label = class_list[class_id]
                else:
                    logger.warning(
                        "Box class ID %s is out of range 0-%s: Could not infer class name",
                        class_id,
                        len(class_list),
                    )
            elif class_id is None and label is not None:
                try:
                    class_id = class_list.index(label)
                except ValueError:
                    logger.warning(
                        "Box class name '%s' not in provided list: Could not infer class ID"
                    )

        return cls(
            top=obj["top"],
            left=obj["left"],
            height=obj["height"],
            width=obj["width"],
            label=label,
            class_id=class_id,
        )
