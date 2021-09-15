# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Lambda function to resume a Step Functions flow on completion of an A2I human loop

How to use:
- Use a lambda:invoke.waitForTaskToken task Lambda to start your human loop
- Add a `taskToken` field to your task input when starting the loop, containing your SFn token
- Use an S3 subscription to trigger this Lambda when objects are created in your A2I output bucket

This function's implementation is dependent on the expected structure of the review task outputs
and the target post-review model result structure - so you'll need to modify it if changing the
task.
"""

# Python Built-Ins:
import json
import logging

# External Dependencies:
import boto3


logger = logging.getLogger()
logger.setLevel(logging.INFO)
s3 = boto3.resource("s3")
sfn = boto3.client("stepfunctions")


class MalformedReviewResponse(ValueError):
    """Returned to SFN when review completed with bad output format (e.g. incompatible workflow)"""

    pass


class ReviewFailed(ValueError):
    """Returned to SFN when review cycle completed unhealthily (e.g. with no human responses)"""

    pass


def handler(event, context):
    """S3 bucket subscription Lambda to pick up and continue processing flow for A2I review results

    Triggered by uploads to the S3 human reviews bucket.

    Input events should be formatted as per S3 notifications - described here:
    https://docs.aws.amazon.com/lambda/latest/dg/with-s3.html
    """
    # Notifications come through in batches
    for record in event["Records"]:
        timestamp = record["eventTime"]
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]
        if key.endswith(".json"):
            print(f"Processing {timestamp} | s3://{bucket}/{key}")
        else:
            print(f"Skipping (not .json) {timestamp} | s3://{bucket}/{key}")
            continue

        # Need to load the result file to extract the Step Functions task token:
        result_response = s3.Object(bucket, key).get()
        result = json.loads(result_response["Body"].read())

        task_input = result["inputContent"]
        task_token = task_input.get("TaskToken")
        if not task_token:
            print(f"WARNING: Missing task token, ignoring result")
            continue
        try:
            n_human_answers = len(result.get("humanAnswers", []))
            if not n_human_answers:
                raise ReviewFailed("A2I review finished with no human responses")
            elif n_human_answers > 1:
                raise NotImplementedError(
                    "Consolidation is only implemented for 1:1 reviews, but A2I response yielded "
                    f"{n_human_answers} human answers. (s3://{bucket}/{key})"
                )

            human_answer = result["humanAnswers"][0]
            reviewed_result = {
                "Fields": {},
                "Review": {
                    "WorkerId": human_answer["workerId"],
                    "TimeSpentInSeconds": human_answer["timeSpentInSeconds"],
                },
            }
            # Pre-consolidate inputs 'field-multival-[\d+]-{name}' to map field names and sorting
            review_multival_inputs = [
                {"InputName": k}
                for k in human_answer["answerContent"]
                if k.startswith("field-multival-")
            ]
            for inp in review_multival_inputs:
                parted = inp["InputName"][len("field-multival-") :].partition("-")
                inp["SortOrder"] = int(parted[0])
                inp["FieldName"] = parted[2]
            review_multival_inputs_sorted = sorted(
                review_multival_inputs,
                key=lambda inp: inp["SortOrder"],
            )

            for field_name, field_input in task_input["ModelResult"]["Fields"].items():
                review_presence = (
                    human_answer["answerContent"]
                    .get(f"field-present-{field_name}", {})
                    .get("checked")
                )
                review_value = human_answer["answerContent"].get(f"field-value-{field_name}")
                is_multivalued = bool("Values" in field_input)
                review_multivalues = [
                    human_answer["answerContent"][inp["InputName"]]
                    for inp in review_multival_inputs_sorted
                    if inp["FieldName"] == "field_name"
                ]
                is_field_review_found = (
                    review_presence == False
                    or (is_multivalued and len(review_multivalues))
                    or (review_value and not is_multivalued)
                )
                if is_multivalued:
                    is_exact_match = True
                    try:
                        for checked_value in review_multivalues:
                            _ = next(
                                v for v in field_input["Values"] if v["Value"] == checked_value
                            )
                        for model_value in field_input["Values"]:
                            _ = next(v for v in review_multivalues if v == model_value["Value"])
                    except StopIteration:
                        is_exact_match = False
                else:
                    is_exact_match = review_value == field_input["Value"]

                if not is_field_review_found:
                    # Review didn't seem to cover the target field - pass it through
                    field_output = field_input
                else:
                    # Pass through some input attributes, replacing others with the review results:
                    field_output = {}
                    for input_k, input_v in field_input.items():
                        if input_k == "Confidence":
                            field_output[input_k] = 1.0
                        elif input_k == "Detections":
                            # If reviewer edited the value(s), we have no way to trace them back to
                            # Textract blocks:
                            field_output[input_k] = input_v if is_exact_match else []
                        elif input_k == "Value":
                            field_output[input_k] = review_value
                        elif input_k == "Values":
                            if is_exact_match:
                                field_output[input_k] = [
                                    {
                                        "Confidence": 1.0,
                                        "Detections": v["Detections"],
                                        "Value": v["Value"],
                                    }
                                    for v in input_v
                                ]
                            else:
                                field_output[input_k] = [
                                    {
                                        "Confidence": 1.0,
                                        "Detections": [],
                                        "Value": v,
                                    }
                                    for v in review_multivalues
                                ]
                        else:
                            # Pass through e.g. ClassId, NumDetectedValues, etc
                            field_output[input_k] = input_v

                reviewed_result["Fields"][field_name] = field_output

            reviewed_result["Confidence"] = min(
                f.get("Confidence", 0) for f in reviewed_result["Fields"].values()
            )
            sfn.send_task_success(
                taskToken=task_token,
                output=json.dumps(reviewed_result),
            )
            logger.info(
                f"Successfully completed review: s3://{bucket}/{key} with token {task_token}",
            )
            print("Notified task complete")
        except KeyError as ke:
            sfn.send_task_failure(
                taskToken=task_token,
                error="MalformedReviewResponse",
                cause=f"Missing field: {str(ke)}",
            )
            logger.exception("Notified task failed - malformed input")
        except Exception as e:
            # Like the default direct-to-Lambda integration, we'll return the Python exception type
            # name and the message:
            sfn.send_task_failure(
                taskToken=task_token,
                error=type(e).__name__,
                cause=str(e),
            )
            logger.exception("Notified task failed")

    return {
        "statusCode": 200,
        "body": json.dumps("Success"),
    }
