# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Lambda to call the SageMaker OCR enrichment model

Properties of the input event are forwarded to boto3 SageMakerRuntime client's invoke_endpoint
method, with the exceptions that;

- If 'EndpointName' is not provided, it's read from environment variable or SSM parameter
- If 'Body' is an object type other than a string, it's automatically JSON stringified for you.
  and 'ContentType' defaulted to 'application/json' if not explicitly set.
- If 'Accept' is not explicitly set, it's defaulted to 'application/json'

The function returns the response body from the endpoint.
"""

# Python Built-Ins:
import json
import logging
import os

# External Dependencies:
import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)
s3 = boto3.resource("s3")
smruntime = boto3.client("sagemaker-runtime")
ssm = boto3.client("ssm")

DEFAULT_ENDPOINT_NAME = os.environ.get("DEFAULT_ENDPOINT_NAME")
DEFAULT_ENDPOINT_NAME_PARAM = os.environ.get("DEFAULT_ENDPOINT_NAME_PARAM")


class MalformedRequest(ValueError):
    pass


def handler(event, context):
    logger.debug("Received event %s", event)
    if "Body" not in event:
        raise MalformedRequest("Input must include 'Body' string or object")

    if "EndpointName" not in event:
        if DEFAULT_ENDPOINT_NAME:
            event["EndpointName"] = DEFAULT_ENDPOINT_NAME
        elif DEFAULT_ENDPOINT_NAME_PARAM:
            event["EndpointName"] = ssm.get_parameter(Name=DEFAULT_ENDPOINT_NAME_PARAM,)[
                "Parameter"
            ]["Value"]
        else:
            raise MalformedRequest(
                "Input must include 'EndpointName' if neither DEFAULT_ENDPOINT_NAME nor "
                "DEFAULT_ENDPOINT_NAME_PARAM environment variables are set"
            )

    # Convert body if JSON:
    if not isinstance(event["Body"], str):
        event["Body"] = json.dumps(event["Body"])
        if "ContentType" not in event:
            event["ContentType"] = "application/json"

    if "Accept" not in event:
        event["Accept"] = "application/json"

    return json.loads(smruntime.invoke_endpoint(**event)["Body"].read())
