# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Custom CloudFormation Resource to kick off CodeBuild project builds

This custom resource expects a 'ProjectName' property, and will simply kick off a run of that AWS
CodeBuild Project on creation. It doesn't wait for the run to complete successfully, and it doesn't
do anything on resource UPDATE/DELETE.
"""
# Python Built-Ins:
import logging
import traceback

# External Dependencies:
import boto3
import cfnresponse

codebuild = boto3.client("codebuild")


def lambda_handler(event, context):
    try:
        request_type = event["RequestType"]
        if request_type == "Create":
            handle_create(event, context)
        elif request_type == "Update":
            handle_update(event, context)
        elif request_type == "Delete":
            handle_delete(event, context)
        else:
            cfnresponse.send(
                event,
                context,
                cfnresponse.FAILED,
                {},
                error=f"Unsupported CFN RequestType '{request_type}'",
            )
    except Exception as e:
        logging.error("Uncaught exception in CFN custom resource handler - reporting failure")
        traceback.print_exc()
        cfnresponse.send(
            event,
            context,
            cfnresponse.FAILED,
            {},
            error=str(e),
        )
        raise e


def handle_create(event, context):
    logging.info("**Received create request")
    resource_config = event["ResourceProperties"]
    logging.info("**Running CodeBuild Job")
    result = codebuild.start_build(
        projectName=resource_config["ProjectName"],
    )
    cfnresponse.send(
        event,
        context,
        cfnresponse.SUCCESS,
        {},
        physicalResourceId=result["build"]["arn"],
    )


def handle_delete(event, context):
    logging.info("**Received delete event - no-op")
    cfnresponse.send(
        event,
        context,
        cfnresponse.SUCCESS,
        {},
        physicalResourceId=event["PhysicalResourceId"],
    )


def handle_update(event, context):
    logging.info("**Received update event - no-op")
    cfnresponse.send(
        event,
        context,
        cfnresponse.SUCCESS,
        {},
        physicalResourceId=event["PhysicalResourceId"],
    )
