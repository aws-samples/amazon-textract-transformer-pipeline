# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CDK constructs and utilities for working with Amazon SageMaker
"""
# External Dependencies:
from aws_cdk import Stack
from aws_cdk.aws_s3 import Bucket
from constructs import Construct

# Local Dependencies:
from .model_deployment import (
    EndpointAutoscaler,
    SageMakerAsyncInferenceConfig,
    SageMakerAutoscalingRole,
    SageMakerCustomizedDLCModel,
    SageMakerDLCBasedImage,
    SageMakerDLCSpec,
    SageMakerEndpointExecutionRole,
    SageMakerModelDeployment,
)
from .sagemaker_sfn import SageMakerCallerFunction, SageMakerSSMStep


def get_sagemaker_default_bucket(scope: Construct) -> Bucket:
    """Generate a CDK S3.Bucket construct for the (assumed pre-existing) SageMaker Default Bucket"""
    stack = Stack.of(scope)
    return Bucket.from_bucket_arn(
        scope,
        "SageMakerDefaultBucket",
        f"arn:{stack.partition}:s3:::sagemaker-{stack.region}-{stack.account}",
    )
