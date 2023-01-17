# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Annotation consolidation Lambda for BBoxes+transcriptions in SageMaker Ground Truth
"""
# Python Built-Ins:
import json
import logging
from typing import List, Optional

# External Dependencies:
import boto3  # AWS SDK for Python

# Set up logger before local imports:
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Local Dependencies:
from data_model import SMGTWorkerAnnotation  # Custom task data model (edit if needed!)
from smgt import (  # Generic SageMaker Ground Truth parsers/utilities
    ConsolidationRequest,
    ObjectAnnotationResult,
    PostConsolidationDatum,
)


s3 = boto3.client("s3")


def consolidate_object_annotations(
    object_data: ObjectAnnotationResult,
    label_attribute_name: str,
    label_categories: Optional[List[str]] = None,
) -> PostConsolidationDatum:
    """Consolidate the (potentially multiple) raw worker annotations for a dataset object

    TODO: Actual consolidation/reconciliation of multiple labels is not yet supported!

    This function just takes the "first" (not necessarily clock-first) worker's result and outputs
    a warning if others were found.

    Parameters
    ----------
    object_data :
        Object describing the raw annotations and metadata for a particular task in the SMGT job
    label_attribute_name :
        Target attribute on the output object to store consolidated label results (note this may
        not be the *only* attribute set/updated on the output object, hence provided as a param
        rather than abstracted away).
    label_categories :
        Label categories specified when creating the labelling job. If provided, this is used to
        translate from class names to numeric class_id similarly to SMGT's built-in bounding box
        task result.
    """
    warn_msgs: List[str] = []
    worker_anns: List[SMGTWorkerAnnotation] = []
    for worker_ann in object_data.annotations:
        ann_raw = worker_ann.fetch_data()
        worker_anns.append(SMGTWorkerAnnotation.parse(ann_raw, class_list=label_categories))

    if len(worker_anns) > 1:
        warn_msg = (
            "Reconciliation of multiple worker annotations is not currently implemented for this "
            "post-processor. Outputting annotation from worker %s and ignoring labels from %s"
            % (
                object_data.annotations[0].worker_id,
                [a.worker_id for a in object_data.annotations[1:]],
            )
        )
        logger.warning(warn_msg)
        warn_msgs.append(warn_msg)

    consolidated_label = worker_anns[0].to_jsonable()
    if len(warn_msgs):
        consolidated_label["consolidationWarnings"] = warn_msgs

    return PostConsolidationDatum(
        dataset_object_id=object_data.dataset_object_id,
        consolidated_content={
            label_attribute_name: consolidated_label,
            # Note: In our tests it's not possible to add a f"{label_attribute_name}-meta" field
            # here - it gets replaced by whatever post-processing happens, instead of merged.
        },
    )


def handler(event: dict, context) -> List[dict]:
    """Main Lambda handler for consolidation of SMGT worker annotations

    This function receives a batched request to consolidate (multiple?) workers' annotations for
    multiple objects, and outputs the consolidated results per object. For more docs see:

    https://docs.aws.amazon.com/sagemaker/latest/dg/sms-custom-templates-step3-lambda-requirements.html
    """
    logger.info("Received event: %s", json.dumps(event))
    req = ConsolidationRequest.parse(event)
    if req.label_categories and len(req.label_categories) > 0:
        label_cats = req.label_categories
    else:
        logger.warning(
            "Label categories list (see CreateLabelingJob.LabelCategoryConfigS3Uri) was not "
            "provided when creating this job. Post-consolidation outputs will be incompatible with "
            "built-in Bounding Box task, because we're unable to map class names to numeric IDs."
        )
        label_cats = None

    # Loop through the objects in this batch, consolidating annotations for each:
    return [
        consolidate_object_annotations(
            object_data,
            label_attribute_name=req.label_attribute_name,
            label_categories=label_cats,
        ).to_jsonable()
        for object_data in req.fetch_object_annotations()
    ]
