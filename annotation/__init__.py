# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CDK Construct for infrastructure to support data annotation/labelling

Custom SageMaker Ground Truth templates require custom pre-processing and result-consolidation
Lambda functions.
"""
# Python Built-Ins:
import os

# External Dependencies:
from aws_cdk import core as cdk
from aws_cdk.aws_iam import (
    ManagedPolicy,
    Role,
    ServicePrincipal,
)
from aws_cdk.aws_lambda import Runtime as LambdaRuntime
from aws_cdk.aws_lambda_python import PythonFunction
from aws_cdk.aws_s3 import Bucket


PRE_LAMBDA_PATH = os.path.join(os.path.dirname(__file__), "fn-SMGT-Pre")
POST_LAMBDA_PATH = os.path.join(os.path.dirname(__file__), "fn-SMGT-Post")


class AnnotationInfra(cdk.Construct):
    """CDK construct for custom SageMaker Ground Truth annotation task infrastructure"""

    def __init__(self, scope: cdk.Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        self.lambda_role = Role(
            self,
            "SMGT-LambdaRole",
            assumed_by=ServicePrincipal("lambda.amazonaws.com"),
            description="Execution role for SageMaker Ground Truth pre/post processing Lambdas",
            managed_policies=[
                ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole",
                ),
                ManagedPolicy.from_aws_managed_policy_name(
                    "AWSXRayDaemonWriteAccess",
                ),
            ],
        )
        Bucket.from_bucket_arn(
            self,
            "SageMakerDefaultBucket",
            f"arn:aws:s3:::sagemaker-{cdk.Stack.of(self).region}-{cdk.Stack.of(self).account}",
        ).grant_read_write(self.lambda_role)

        self._pre_lambda = PythonFunction(
            self,
            # Include 'LabelingFunction' in the name so the entities with the
            # AmazoSageMakerGroundTruthExecution policy will automatically have access to call it:
            # https://console.aws.amazon.com/iam/home?#/policies/arn:aws:iam::aws:policy/AmazonSageMakerGroundTruthExecution
            "SMGT-LabelingFunction-Pre",
            entry=PRE_LAMBDA_PATH,
            index="main.py",
            handler="handler",
            memory_size=128,
            role=self.lambda_role,
            runtime=LambdaRuntime.PYTHON_3_8,
            timeout=cdk.Duration.seconds(5),
        )
        self._post_lambda = PythonFunction(
            self,
            "SMGT-LabelingFunction-Post",
            entry=POST_LAMBDA_PATH,
            index="main.py",
            handler="handler",
            memory_size=128,
            role=self.lambda_role,
            runtime=LambdaRuntime.PYTHON_3_8,
            timeout=cdk.Duration.seconds(60),
        )

    @property
    def pre_lambda(self):
        return self._pre_lambda

    @property
    def post_lambda(self):
        return self._post_lambda
