# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Lambda to call the (real-time or async) SageMaker enrichment model and handle SNS callbacks

In general, properties of the input event are forwarded to boto3 SageMakerRuntime client's
invoke_endpoint or invoke_endpoint_async method, except that:

- If 'EndpointName' is not provided, it's read from environment variable or SSM parameter.
- If 'Accept' is not explicitly set, it's defaulted to 'application/json'.
- If 'TaskToken' is provided, it's interpreted as a task token to return results to Step Functions
  via async integration.

Checks for SSM EndpointName, and whether an endpoint is asynchronous, are cached for
CACHE_TTL_SECONDS to reduce GetParameter/DescribeEndpoint API volume.

For synchronous endpoints
-------------------------

- If 'Body' is an object type other than a string, it's automatically JSON stringified for you.
  and 'ContentType' defaulted to 'application/json' if not explicitly set.

For asynchronous endpoints
--------------------------

- The SFn TaskToken is passed through using SageMaker's 'CustomAttributes' parameter, so this is
  not available for other purposes.
- If 'Body' (which is not supported for async as the input should be an object on S3) is present,
  the function will try to interpret it (or Body.S3Input) as a pointer to an S3 InputLocation.
- If 'InputLocation' (usually an "s3://..." URI) is a dict with { Bucket, Key } or { S3Uri }, it
  will be converted automatically.

Return Value
------------

result : dict
    For synchronous invocations without a TaskToken, result is the endpoint response body. For SNS
    callback messages, result is a batch of responses from SFn from recording each task result. For
    inference requests when TaskToken is provided, result may be a SFn response (if the endpoint is
    synchronous) Or just the SageMaker InvokeEndpointAsync response (if async).
"""

# Python Built-Ins:
import json
import logging
import os
import sys
import traceback

# External Dependencies:
import boto3
from botocore.config import Config as BotoConfig
from cachetools import cached, TTLCache

logger = logging.getLogger()
logger.setLevel(logging.INFO)
boto_config = BotoConfig(retries={"max_attempts": 5, "mode": "standard"})
s3 = boto3.resource("s3", config=boto_config)
sfn = boto3.client("stepfunctions", config=boto_config)
smclient = boto3.client("sagemaker", config=boto_config)
smruntime = boto3.client("sagemaker-runtime", config=boto_config)
ssm = boto3.client("ssm", config=boto_config)

CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", str(3 * 60)))
DEFAULT_ENDPOINT_NAME = os.environ.get("DEFAULT_ENDPOINT_NAME")
DEFAULT_ENDPOINT_NAME_PARAM = os.environ.get("DEFAULT_ENDPOINT_NAME_PARAM")
SUPPORT_ASYNC_ENDPOINTS = os.environ.get("SUPPORT_ASYNC_ENDPOINTS", "0").lower()
if SUPPORT_ASYNC_ENDPOINTS in ("1", "y", "yes", "true", "t"):
    SUPPORT_ASYNC_ENDPOINTS = True
elif SUPPORT_ASYNC_ENDPOINTS in ("0", "n", "no", "false", "f"):
    SUPPORT_ASYNC_ENDPOINTS = False
else:
    raise ValueError(
        "Could not interpret boolean env var SUPPORT_ASYNC_ENDPOINTS. Got: '%s'"
        % (SUPPORT_ASYNC_ENDPOINTS,)
    )


class MalformedRequest(ValueError):
    pass


class InferenceFailed(ValueError):
    pass


class S3ObjectSpec:
    """Utility class for parsing an S3 location spec from a JSON-able dict

    Shared with SageMaker endpoint inference.py
    """

    bucket: str
    key: str

    def __init__(self, spec: dict):
        if "S3Uri" in spec:
            if not spec["S3Uri"].lower().startswith("s3://"):
                raise ValueError("S3Uri must be a valid 's3://...' URI if provided")
            bucket, _, key = spec["S3Uri"][len("s3://") :].partition("/")
        else:
            bucket = spec.get("Bucket")
            key = spec.get("Key")
        if not (bucket and key and isinstance(bucket, str) and isinstance(key, str)):
            raise MalformedRequest(
                "Must provide an object with either 'S3Uri' or 'Bucket' and 'Key' properties. "
                f"Parsed bucket={bucket}, key={key}"
            )
        self.bucket = bucket
        self.key = key

    @property
    def uri(self):
        return f"s3://{self.bucket}/{self.key}"


@cached(cache=TTLCache(maxsize=64, ttl=CACHE_TTL_SECONDS))
def is_endpoint_async(endpoint_name: str) -> bool:
    """Check whether an SM endpoint is asynchronous (with caching)

    Uses the sagemaker:DescribeEndpoint API, or serves a recent cached result if one is known.
    """
    endpoint_desc = smclient.describe_endpoint(EndpointName=endpoint_name)
    result = "AsyncInferenceConfig" in endpoint_desc
    logger.info(f"SageMaker Endpoint {endpoint_name} IS {'' if result else 'NOT '} asynchronous")
    return result


@cached(cache=TTLCache(maxsize=2, ttl=CACHE_TTL_SECONDS))
def get_endpoint_name_from_ssm() -> str:
    """Fetch configured endpoint name from SSM (with caching)

    Fetches the SageMaker endpoint name from SSM, or serves a recent cached result if one is known.
    """
    result = ssm.get_parameter(Name=DEFAULT_ENDPOINT_NAME_PARAM,)[
        "Parameter"
    ]["Value"]
    logger.info(f"Endpoint name from SSM is: {result}")
    return result


def handler(event: dict, context):
    """Main Lambda handler"""
    logger.info("Received event %s", event)

    # First thing we do is try to retrieve the SFn task token if present, and then wrap remaining
    # steps in a try/except - to ensure we notify SFn of errors promptly rather than timing out.
    task_token = event.get("TaskToken")
    try:
        if "Records" in event:
            # Batched callback of inference results (e.g. from SNS)
            result = {"Records": []}
            for ix, record in enumerate(event["Records"]):
                if "EventSource" not in record:
                    raise MalformedRequest(
                        "Record %s does not specify required field 'EventSource'",
                        ix,
                    )
                # TODO: Support messages via SQS too
                if record["EventSource"] != "aws:sns" or not isinstance(
                    record.get("Sns", {}).get("Message"), str
                ):
                    raise MalformedRequest(
                        "Record %s must have EventSource='aws:sns' and string prop Sns.Message",
                        ix,
                    )
                result["Records"].append(
                    handle_callback(json.loads(record["Sns"]["Message"]), context),
                )
            return result
        else:
            # Not a callback: An actual inference request:
            return handle_request(event, context)
    except Exception as e:
        if task_token:
            return send_error(e, task_token)
        else:
            raise e


def handle_request(event: dict, context):
    """Lambda handler for inference requests"""

    if "EndpointName" not in event:
        if DEFAULT_ENDPOINT_NAME:
            event["EndpointName"] = DEFAULT_ENDPOINT_NAME
        elif DEFAULT_ENDPOINT_NAME_PARAM:
            event["EndpointName"] = get_endpoint_name_from_ssm()
        else:
            raise MalformedRequest(
                "Input must include 'EndpointName' if neither DEFAULT_ENDPOINT_NAME nor "
                "DEFAULT_ENDPOINT_NAME_PARAM environment variables are set"
            )

    if "Accept" not in event:
        event["Accept"] = "application/json"

    if "TaskToken" in event:
        task_token = event.pop("TaskToken")
        if not SUPPORT_ASYNC_ENDPOINTS:
            raise ValueError(
                "event.TaskToken is not supported when SUPPORT_ASYNC_ENDPOINTS env var is false"
            )

        if is_endpoint_async(event["EndpointName"]):
            return prepare_invoke_async(event, task_token)
        else:
            result = prepare_invoke_sync(event)
            send_result(result, task_token)
            return result
    else:
        return prepare_invoke_sync(event)


def prepare_invoke_sync(event: dict) -> dict:
    """Synchronous-endpoint-specific event preparation and invocation

    Modifies event object in-place
    """
    if "Body" not in event:
        raise MalformedRequest("Sync endpoint input must include 'Body' string or object")

    # Convert body if JSON:
    if not isinstance(event["Body"], str):
        event["Body"] = json.dumps(event["Body"])
        if "ContentType" not in event:
            event["ContentType"] = "application/json"

    return json.loads(smruntime.invoke_endpoint(**event)["Body"].read())


def prepare_invoke_async(event: dict, task_token: str) -> dict:
    """Asynchronous-endpoint-specific event preparation and invocation

    Modifies event object in-place
    """
    if "CustomAttributes" in event:
        raise ValueError(
            "Can't specify CustomAttributes in async endpoint mode, because this field is used to "
            "pass SFn task token."
        )
    event["CustomAttributes"] = task_token

    if "Body" in event:
        # Body is not supported by async (should be a pointer to where the input data is on S3),
        # but if the provided Body is obviously an S3 pointer, we'll work with it:
        req_body = event.pop("Body")
        if isinstance(req_body, str):
            req_body_lower = req_body.lower()
            if req_body_lower.startswith("s3://") or req_body_lower.startswith("https://"):
                event["InputLocation"] = req_body
            else:
                try:
                    s3spec = S3ObjectSpec(json.loads(req_body))
                    event["InputLocation"] = s3spec.uri
                except Exception as e:
                    raise ValueError(
                        "event.Body string could not be interpreted as an S3://... input URI for "
                        "async endpoint. Async endpoints only support input from S3."
                    ) from e
        elif isinstance(req_body, dict):
            s3spec = None
            try:
                s3spec = S3ObjectSpec(req_body)
            except Exception as e:
                pass
            if (not s3spec) and "S3Input" in req_body:
                try:
                    s3spec = S3ObjectSpec(req_body["S3Input"])
                    other_keys = [k for k in req_body if k != "S3Input"]
                    if len(other_keys):
                        logger.warning(
                            "Body keys besides S3Input will not be sent to async endpoint: %s",
                            other_keys,
                        )
                except Exception as e:
                    pass
            if not s3spec:
                raise ValueError(
                    "event.Body dict could not be interpreted as an S3 object pointer (with "
                    ".Bucket and .Key or .S3Uri, or an .S3Input object containing same) for async "
                    "endpoint. Async endpoints only support input from S3."
                )

            event["InputLocation"] = s3spec.uri
        else:
            raise ValueError(
                "event.Body could not be interpreted as an S3 object pointer for async endpoint. "
                f"Async endpoints only support input from S3. Got Body type {type(req_body)}"
            )

    if "InputLocation" not in event:
        raise MalformedRequest(
            "When using an async SM endpoint, must provide either event.InputLocation or an "
            "event.Body that points to an input request object on S3."
        )
    elif isinstance(event["InputLocation"], dict):
        try:
            s3spec = S3ObjectSpec(event["InputLocation"])
            event["InputLocation"] = s3spec.uri
        except Exception as e:
            raise ValueError(
                "event.InputLocation must be an s3:// URI string or an object with .Bucket, .Key "
                "or .S3Uri."
            )

    resp = smruntime.invoke_endpoint_async(**event)
    logger.info("Started SageMaker async invocation ID %s", resp["InferenceId"])
    return resp


def handle_callback(event: dict, context):
    """Lambda handler for individual async inference callback messages (e.g. via SNS)"""
    task_token = event.get("requestParameters", {}).get("customAttributes")
    if not task_token:
        raise ValueError("Couldn't determine task token from SNS callback message")

    try:
        req_status = event.get("invocationStatus", "")
        if req_status.lower() != "completed":
            raise InferenceFailed(
                event["failureReason"]
                if "failureReason" in event
                else f"SageMaker callback non-completed status '{req_status}'"
            )

        output_s3uri = event.get("responseParameters", {}).get("outputLocation")
        if not output_s3uri:
            raise ValueError(
                "Couldn't determine responseParameters.outputLocation from SNS callback message"
            )
        output_bucket, _, output_key = output_s3uri[len("s3://") :].partition("/")
        return send_result(
            {
                "Bucket": output_bucket,
                "Key": output_key,
                "S3Uri": output_s3uri,
            },
            task_token,
        )
    except Exception as e:
        return send_error(e, task_token)


def send_result(
    result: dict,
    sfn_task_token: str = None,
) -> dict:
    """Send result object to Step Functions if task token is available, else just return it"""
    if sfn_task_token is None:
        return result
    else:
        return sfn.send_task_success(taskToken=sfn_task_token, output=json.dumps(result))


def send_error(err: Exception, task_token: str = None):
    """Send error to Step Functions if task token is available, else re-raise it"""
    if task_token:
        logger.exception("Reporting unhandled exception to Step Functions")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        return sfn.send_task_failure(
            taskToken=task_token,
            error=exc_type.__name__,
            cause=(f"{exc_value}\n" + "".join(traceback.format_stack()))[:256],
        )
    else:
        raise err
