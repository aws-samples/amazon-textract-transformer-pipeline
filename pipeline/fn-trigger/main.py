# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Lambda function to trigger the OCR state machine from an S3 object upload notification
"""
# Python Built-Ins:
from datetime import datetime
import json
import logging
import os
from packaging import version
import re
from typing import List
from urllib.parse import unquote_plus

# External Dependencies:
import boto3  # AWS SDK for Python

logger = logging.getLogger()
logger.setLevel(logging.INFO)
sfn = boto3.client("stepfunctions")

STATE_MACHINE_ARN = os.environ.get("STATE_MACHINE_ARN")
S3_EVENT_STRUCTURE_MAJOR = 2


class MalformedRequest(ValueError):
    pass


class S3Event:
    """Model for an individual record within an S3 notification"""

    bucket: str
    id: str
    key: str

    def __init__(self, record: dict):
        if not record:
            raise MalformedRequest(f"Empty record in S3 notification: {record}")

        self.event_version = record.get("eventVersion")
        if version.parse(self.event_version).major != S3_EVENT_STRUCTURE_MAJOR:
            raise NotImplementedError(
                f"S3 event version {self.event_version} is not supported by this solution."
            )

        self.bucket = record.get("s3", {}).get("bucket", {}).get("name")
        if not self.bucket:
            raise MalformedRequest(f"s3.bucket.name not found in notification: {record}")

        # S3 object notifications quote object key spaces wih '+'. Can undo as follows:
        self.key = unquote_plus(record.get("s3", {}).get("object", {}).get("key"))
        if not self.key:
            raise MalformedRequest(f"s3.object.key not found in notification: {record}")

        # A Step-Functions-compatible event ID with timestamp and filename:
        self.id = re.sub(
            r"[\-]{2,}",
            "-",  # Reduce consecutive hyphens
            re.sub(
                r"[\s<>\{\}\[\]\?\*\"#%\\\^\|\~`\$&,;:/\u0000-\u001F\u007F-\u009F]+",
                "-",  # Replace special chars in filename with hyphens
                "-".join(
                    (
                        # ISO timestamp (millisecond precision) of event:
                        record.get("eventTime", datetime.now().isoformat()),
                        # Filename of document:
                        self.key.rpartition("/")[2],
                    )
                ),
            ),
        )[:80]


class S3Notification:
    """Model for an S3 event notification comprising multiple records"""

    events: List[S3Event]
    parse_errors: List[Exception]

    def __init__(self, event: dict):
        records = event.get("Records")
        if not records:
            raise MalformedRequest("Couldn't find 'Records' array in input event")
        elif not len(records):
            raise MalformedRequest("No Records to process in input event")
        self.events = []
        self.parse_errors = []
        for record in records:
            try:
                self.events.append(S3Event(record))
            except Exception as err:
                logger.exception("Failed to parse S3 notification record")
                self.parse_errors.append(err)


def handler(event: dict, context):
    """Trigger the OCR pipeline state machine in response to an S3 event notification"""
    s3notification = S3Notification(event)

    for record in s3notification.events:
        sfn_input = {
            "Input": {
                "Bucket": record.bucket,
                "Key": record.key,
            },
        }

        sfn.start_execution(
            stateMachineArn=STATE_MACHINE_ARN,
            name=record.id,
            input=json.dumps(sfn_input),
        )
        logger.info(f"Started SFn execution {record.id} from s3://${record.bucket}/{record.key}")
