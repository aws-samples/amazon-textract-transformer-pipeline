# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Annotation consolidation Lambda for BBoxes+transcriptions in SageMaker Ground Truth
"""
# Python Built-Ins:
import json
import logging
import re
from urllib.parse import urlparse

# External Dependencies:
import boto3  # AWS SDK for Python

logger = logging.getLogger()
logger.setLevel(logging.INFO)
s3 = boto3.client("s3")


def handler(event, context):
    consolidated_labels = []

    parsed_url = urlparse(event["payload"]["s3Uri"])
    logger.info("Consolidating labels from %s", event["payload"]["s3Uri"])
    textFile = s3.get_object(Bucket=parsed_url.netloc, Key=parsed_url.path[1:])
    filecont = textFile["Body"].read()
    annotations = json.loads(filecont)

    for dataset in annotations:
        dataset_worker_anns = []
        consolidated_label = {
            "workerAnnotations": dataset_worker_anns,
        }
        dataset_warnings = []

        label = {
            "datasetObjectId": dataset["datasetObjectId"],
            "consolidatedAnnotation": {
                "content": {
                    event["labelAttributeName"]: consolidated_label,
                },
            },
        }

        for annotation in dataset["annotations"]:
            ann_raw = json.loads(annotation["annotationData"]["content"])
            ann_data = json.loads(annotation["annotationData"]["content"])  # (Deep clone of raw)
            ann_data["workerId"] = annotation["workerId"]
            # Find the unique OCR annotation IDs:
            ann_ocr_ids = set(
                map(
                    lambda m: m.group(1),
                    filter(
                        lambda m: m,
                        map(
                            lambda key: re.match(r"ocr-(.*)-[a-z]+", key, flags=re.IGNORECASE),
                            ann_raw.keys(),
                        ),
                    ),
                ),
            )
            # Normalize the OCR labels for this annotation:
            ocr_ann_data = []
            ann_data["ocrAnnotations"] = ocr_ann_data
            for ocr_id in ann_ocr_ids:
                meta_field_key = f"ocr-{ocr_id}-meta"
                if meta_field_key in ann_data:
                    ocr_datum = json.loads(ann_data[meta_field_key])
                    del ann_data[meta_field_key]
                else:
                    ocr_datum = {}
                ocr_datum["annotationId"] = ocr_id

                # Consolidate the field's status from (potentially missing/inconsistent) radios:
                ocr_statuses = ("correct", "unclear", "wrong")
                ocr_status_fields = [f"ocr-{ocr_id}-{s}" for s in ocr_statuses]
                unknown_statuses = [
                    s for ix, s in enumerate(ocr_statuses) if ocr_status_fields[ix] not in ann_data
                ]
                selected_statuses = [
                    s
                    for ix, s in enumerate(ocr_statuses)
                    if ann_data.get(ocr_status_fields[ix], {}).get("on")
                ]
                if len(selected_statuses) >= 1:
                    ocr_datum["status"] = selected_statuses[0]
                else:
                    dataset_warnings.append(
                        f"Missing correct/unclear/wrong status for OCR field {ocr_id}",
                    )
                if len(selected_statuses) > 1:
                    dataset_warnings.append(
                        "OCR field {} tagged to multiple statuses {}: Taking first value".format(
                            ocr_id,
                            selected_statuses,
                        )
                    )
                if len(unknown_statuses):
                    dataset_warnings.append(
                        "".join(
                            "Could not determine whether the following statuses were selected ",
                            "for OCR field {}: {}",
                        ).format(
                            ocr_id,
                            unknown_statuses,
                        )
                    )
                for key in ocr_status_fields:
                    if key in ann_data:
                        del ann_data[key]

                # Load in the correction text, if provided:
                correction_field_key = f"ocr-{ocr_id}-override"
                if correction_field_key in ann_data:
                    # Ignore correction if 'wrong' was not selected:
                    if "wrong" in selected_statuses:
                        ocr_datum["correction"] = ann_data[correction_field_key]
                    # Tidy up the raw field regardless:
                    del ann_data[correction_field_key]

                ocr_ann_data.append(ocr_datum)
            dataset_worker_anns.append(ann_data)

        if len(dataset_warnings):
            consolidated_label["consolidationWarnings"] = dataset_warnings
        if len(dataset_worker_anns):
            # Take first annotation as 'consolidated' value:
            for key in dataset_worker_anns[0]:
                if key not in consolidated_label:
                    consolidated_label[key] = dataset_worker_anns[0][key]
        consolidated_labels.append(label)

    return consolidated_labels
