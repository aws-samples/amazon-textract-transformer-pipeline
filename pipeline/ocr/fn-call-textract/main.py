# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Lambda to run Amazon Textract OCR (from Step Function)

This function handles *both* the initial request to Amazon Textract, and (in case the asynchronous
APIs are being used) the callback when the job is completed.
"""

# Python Built-Ins:
import hashlib
import io
import json
import logging
import os
import sys
import time
import traceback

# External Dependencies:
import boto3
from botocore.config import Config as BotoConfig

logger = logging.getLogger()
logger.setLevel(logging.INFO)
boto_config = BotoConfig(retries={"max_attempts": 10, "mode": "standard"})
ddb = boto3.resource("dynamodb", config=boto_config)
s3 = boto3.resource("s3", config=boto_config)
sfn = boto3.client("stepfunctions", config=boto_config)
textract = boto3.client("textract", config=boto_config)


def boolean_env_var(raw: str) -> bool:
    raw_lower = str(raw).lower()
    if raw_lower in ("1", "true", "y", "yes"):
        return True
    elif raw_lower in ("0", "false", "n", "no"):
        return False
    else:
        raise ValueError(f"Unrecognised boolean env var value '{raw}'")


is_textract_sync = os.environ.get("TEXTRACT_INTEGRATION_TYPE", "ASYNC").upper() in (
    "SYNC",
    "SYNCHRONOUS",
)
default_textract_features = list(
    filter(lambda f: f, os.environ.get("DEFAULT_TEXTRACT_FEATURES", "").upper().split(","))
)
# Set true to always use AnalyzeDoc APIs (not DetectText), even if no extra features specified:
force_analyze_apis = boolean_env_var(os.environ.get("FORCE_ANALYZE_APIS", "no"))
callback_sns_role_arn = os.environ.get("CALLBACK_SNS_ROLE_ARN")
callback_sns_topic_arn = os.environ.get("CALLBACK_SNS_TOPIC_ARN")
ddb_state_cache_table_name = os.environ.get("DDB_STATE_CACHE_TABLE")

TEXTRACT_JOBTAG_MAX_LEN = 64
STATE_CACHE_TTL_SECS = 24 * 60 * 60  # 12hrs
GET_DOCUMENT_ANALYSIS_TPS_LIMIT = 2
GET_DOCUMENT_TEXT_DETECTION_TPS_LIMIT = 2

ddb_state_cache_table = (
    ddb.Table(ddb_state_cache_table_name) if ddb_state_cache_table_name else None
)


class MalformedRequest(ValueError):
    pass


def handler(event, context):
    logger.info("Received event: %s", event)
    task_token = event.get("TaskToken")
    try:
        if "Input" in event:
            return handle_request(event, context)
        elif "JobId" in event:
            # Direct callback
            return handle_callback(event, context)
        elif "Records" in event:
            # Batched callback (e.g. from SNS)
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
            raise MalformedRequest(
                "Event did not contain 'Input' (for new analysis request), 'JobId' (for direct "
                "callback) or 'Records' (for callback from SNS/SQS). Please check your input "
                "payload."
            )
    except Exception as e:
        return send_error(e, task_token)


def handle_request(event, context):
    try:
        srcbucket = event["Input"]["Bucket"]
        srckey = event["Input"]["Key"]
        task_token = event["TaskToken"]
        textract_features = event["Output"].get("Features", default_textract_features)
        output_type = event["Output"].get("Type", "s3")
        output_type_lower = output_type.lower()
        if output_type_lower == "inline":
            destbucket = None
            destkey = None
            pass  # No other config to collect
        elif output_type_lower == "s3":
            destbucket = event["Output"].get("Bucket", srcbucket)
            if event["Output"].get("Key"):
                destkey = event["Output"]["Key"]
            else:
                prefix = event["Output"].get("Prefix", "")
                if prefix and not prefix.endswith("/"):
                    prefix += "/"
                destkey = "".join([prefix, srckey, ".textract.json"])
        else:
            raise MalformedRequest(
                f"Unknown output integration type '{output_type}': Expected 'Inline' or 'S3'"
            )
    except KeyError as ke:
        raise MalformedRequest(f"Missing field {ke}, please check your input payload")

    input_doc = {
        "S3Object": {
            "Bucket": srcbucket,
            "Name": srckey,
        },
    }
    if is_textract_sync:
        if len(textract_features) or force_analyze_apis:
            result = textract.analyze_document(
                Document=input_doc,
                FeatureTypes=textract_features,
            )
        else:
            result = textract.detect_document_text(
                Document=input_doc,
            )

        return send_result(
            textract_result=result,
            sfn_task_token=task_token,
            dest_bucket=destbucket,
            dest_key=destkey,
        )
    else:
        # ClientRequestToken allows idempotency in case of retries - but only up to 64 chars (so
        # we can't use the TaskToken). Different use cases might have different idempotency needs,
        # but remember that Textract won't re-trigger an SNS notification for a job it remembers.
        # By default we'll enforce idempotency by source location + target location (or current
        # timestamp if target location not provided), and provide an additional `IdempotencySalt`
        # escape hatch in case users need to force re-processing:
        urihash = hashlib.sha256()
        urihash.update(f"s3://{srcbucket}/{srckey}".encode("utf-8"))
        if destbucket:
            urihash.update(f"s3://{destbucket}/{destkey}".encode("utf-8"))
            if "IdempotencySalt" in event:
                urihash.update(event["IdempotencySalt"].encode("utf-8"))
        else:
            if "IdempotencySalt" in event:
                urihash.update(event["IdempotencySalt"].encode("utf-8"))
            else:
                urihash.update(f"{time.time()}".encode("utf-8"))
        # TODO: Any way we could catch an idempotency issue and pro-actively return result?
        start_params = {
            "DocumentLocation": input_doc,
            "ClientRequestToken": urihash.hexdigest(),
            # "JobTag": "Howdy",
            "NotificationChannel": {
                "RoleArn": callback_sns_role_arn,
                "SNSTopicArn": callback_sns_topic_arn,
            },
        }

        logger.info("Textract params: %s (features = %s)", start_params, textract_features)
        if len(textract_features) or force_analyze_apis:
            job = textract.start_document_analysis(
                FeatureTypes=textract_features,
                **start_params,
            )
        else:
            job = textract.start_document_text_detection(**start_params)
        logger.info(
            "Started async job %s for s3://%s/%s\n%s",
            job.get("JobId"),
            srcbucket,
            srckey,
            job,
        )
        cache_data = {
            "TextractJobId": job["JobId"],
            "SFnTaskToken": task_token,
            "OutputType": output_type_lower,
        }
        if STATE_CACHE_TTL_SECS:
            cache_data["ExpiresAt"] = int(time.time()) + STATE_CACHE_TTL_SECS
        if output_type_lower == "s3":
            cache_data["OutputS3Bucket"] = destbucket
            cache_data["OutputS3Key"] = destkey
        ddb_state_cache_table.put_item(
            Item=cache_data,
            ReturnValues="NONE",
        )
        return cache_data


def handle_callback(event, context):
    """Handle an asynchronous Textract job completion callback datum from SNS

    Retrieve job metadata from the DynamoDB cache and thus return the result to the requesting Step
    Functions execution, either inline or via S3.
    """
    task_desc = ddb_state_cache_table.get_item(
        Key={"TextractJobId": event["JobId"]},
    )["Item"]
    task_token = task_desc["SFnTaskToken"]
    logger.info("Retrieved task token for Textract job %s", event["JobId"])

    try:
        result = fetch_textract_result(event["JobId"], event.get("API"))
        # Don't need to delete the DDB cache item because it has an expiry anyway
        return send_result(
            textract_result=result,
            sfn_task_token=task_token,
            dest_bucket=task_desc.get("OutputS3Bucket"),
            dest_key=task_desc.get("OutputS3Key"),
        )
    except Exception as e:
        return send_error(e, task_token)


def fetch_textract_result(job_id: str, start_api: str = None):
    """Fetch a (potentially paginated) Textract async job result into memory"""
    next_token = None
    result = {}
    retrieved_all = False
    while not retrieved_all:
        req = {"JobId": job_id}
        if next_token:
            req["NextToken"] = next_token
        last_req_timestamp = time.time()
        part = (
            textract.get_document_text_detection(**req)
            if start_api == "StartDocumentTextDetection"
            else textract.get_document_analysis(**req)
        )
        for key in part:
            if key == "NextToken":
                continue
            elif key in result and isinstance(result[key], list):
                result[key] += part[key]
            else:
                result[key] = part[key]
        next_token = part.get("NextToken")
        if next_token is None:
            retrieved_all = True
        else:
            next_req_timestamp = last_req_timestamp + 1 / (
                GET_DOCUMENT_TEXT_DETECTION_TPS_LIMIT
                if start_api == "StartDocumentTextDetection"
                else GET_DOCUMENT_ANALYSIS_TPS_LIMIT
            )
            time.sleep(max(0, next_req_timestamp - time.time()))
    return result


def send_result(
    textract_result: dict,
    sfn_task_token: str = None,
    dest_bucket: str = None,
    dest_key: str = None,
):
    if dest_bucket is None:
        output = textract_result
    else:
        resio = io.BytesIO(json.dumps(textract_result).encode("utf-8"))
        s3.Bucket(dest_bucket).upload_fileobj(resio, dest_key)
        output = {
            "Bucket": dest_bucket,
            "Key": dest_key,
            "S3Uri": f"s3://{dest_bucket}/{dest_key}",
        }

    if sfn_task_token is None:
        return output
    else:
        return sfn.send_task_success(taskToken=sfn_task_token, output=json.dumps(output))


def send_error(err: Exception, task_token: str = None):
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
