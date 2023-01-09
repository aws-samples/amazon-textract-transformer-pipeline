# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Utils to normalize detected entity text by calling SageMaker sequence-to-sequence model endpoints

`normalizer_endpoint` on a FieldConfiguration is assumed to be a deployed real-time SageMaker
endpoint that accepts batched 'application/json' requests of structure:
`{"inputs": ["list", "of", "strings"]}`, and returns 'application/json' responses of structure:
`{"generated_text": ["corresponding", "result", "strings"]}`
"""
# Python Built-Ins:
import json
from logging import getLogger
from typing import Dict, List, Sequence

# External Dependencies:
import boto3  # General-purpose AWS SDK for Python

# Local Dependencies:
from .config import FieldConfiguration
from .extract import EntityDetection

logger = getLogger("postproc")
smruntime = boto3.client("sagemaker-runtime")


def normalize_detections(
    detections: Sequence[EntityDetection],
    entity_config: Sequence[FieldConfiguration],
) -> None:
    """Normalize detected entities in-place via batched requests to SageMaker normalizer endpoint(s)

    Due to the high likelihood of one document featuring multiple matches of the same text for the
    same entity class, we de-duplicate requests by target endpoint and input text - and duplicate
    the result across all linked detections.
    """
    entity_config_by_clsid = {c.class_id: c for c in entity_config if not c.ignore}

    # Batched normalization requests:
    # - By target endpoint name
    # - By input text (after adding prompt prefix)
    # - List of which detections (indexes) correspond to the request
    norm_requests: Dict[str, Dict[str, List[int]]] = {}

    # Collect required normalization requests from the detections:
    for ixdet, detection in enumerate(detections):
        config = entity_config_by_clsid.get(detection.cls_id)
        if not config:
            continue  # Ignore any detections in non-configured classes
        if not config.normalizer_endpoint:
            continue  # This entity class configuration has no normalizer
        if config.normalizer_endpoint not in norm_requests:
            norm_requests[config.normalizer_endpoint] = {}

        norm_input_text = config.normalizer_prompt + detection.text
        if norm_input_text in norm_requests[config.normalizer_endpoint]:
            norm_requests[config.normalizer_endpoint][norm_input_text].append(ixdet)
        else:
            norm_requests[config.normalizer_endpoint][norm_input_text] = [ixdet]

    # Call out to the SageMaker endpoints and update the detections with the results:
    for endpoint_name in norm_requests:
        req_dict = norm_requests[endpoint_name]
        input_texts = [k for k in req_dict]
        try:
            norm_resp = smruntime.invoke_endpoint(
                EndpointName=endpoint_name,
                Body=json.dumps(
                    {
                        "inputs": input_texts,
                    }
                ),
                ContentType="application/json",
                Accept="application/json",
            )
            # Response should be JSON dict containing list 'generated_text' of outputs:
            output_texts = json.loads(norm_resp["Body"].read())["generated_text"]
        except Exception:
            # Log the failure, but continue on:
            logger.exception(
                "Entity normalization call failed: %s texts to endpoint '%s'",
                len(input_texts),
                endpoint_name,
            )
            continue

        for ixtext, output in enumerate(output_texts):
            for ixdetection in req_dict[input_texts[ixtext]]:
                detections[ixdetection].normalize(output)

    # Return nothing to explicitly indicate that detections are modified in-place
    return
