# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""A minimal Lambda function for pre-processing SageMaker Ground Truth custom annotation tasks
"""
import logging

logger = logging.getLogger()


def handler(event, context):
    logger.debug("Got event: %s", event)
    result = {"taskInput": event["dataObject"]}
    logger.debug("Returning result: %s", result)
    return result
