# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CDK Construct for infrastructure to support data annotation/labelling

Custom SageMaker Ground Truth templates require custom pre-processing and result-consolidation
Lambda functions.
"""
# Python Built-Ins:
import os
from typing import List

# External Dependencies:
from aws_cdk import Duration, Stack
from aws_cdk.aws_iam import (
    Effect,
    ManagedPolicy,
    PolicyDocument,
    PolicyStatement,
    Role,
    ServicePrincipal,
)
from aws_cdk.aws_lambda import Runtime as LambdaRuntime
from aws_cdk.aws_lambda_python_alpha import PythonFunction
from aws_cdk.aws_s3 import Bucket
from constructs import Construct


PRE_LAMBDA_PATH = os.path.join(os.path.dirname(__file__), "fn-SMGT-Pre")
POST_LAMBDA_PATH = os.path.join(os.path.dirname(__file__), "fn-SMGT-Post")


class AnnotationInfra(Construct):
    """CDK construct for custom SageMaker Ground Truth annotation task infrastructure"""

    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        self.sm_image_build_role = Role(
            self,
            "SMImageBuildRole",
            assumed_by=ServicePrincipal("codebuild.amazonaws.com"),
            description=(
                "CodeBuild Role for data scientist to build ECR containers for OCR preprocessing"
            ),
            inline_policies={
                "OCRPipelineImageBuild": PolicyDocument(
                    # Scoped down from permissions defined by the sagemaker-studio-image-build-cli:
                    # https://github.com/aws-samples/sagemaker-studio-image-build-cli
                    statements=[
                        PolicyStatement(
                            sid="CreateCodeBuildLogStreams",
                            actions=["logs:CreateLogStream"],
                            effect=Effect.ALLOW,
                            resources=[
                                "arn:aws:logs:*:*:log-group:/aws/codebuild/sagemaker-studio*",
                            ],
                        ),
                        PolicyStatement(
                            sid="CreateLogGroups",
                            actions=["logs:CreateLogGroup"],
                            effect=Effect.ALLOW,
                            resources=["*"],
                        ),
                        PolicyStatement(
                            sid="CodeBuildLogEvents",
                            actions=[
                                "logs:GetLogEvents",
                                "logs:PutLogEvents",
                            ],
                            effect=Effect.ALLOW,
                            resources=[
                                "arn:aws:logs:*:*:log-group:/aws/codebuild/sagemaker-studio*:log-stream:*",
                            ],
                        ),
                        PolicyStatement(
                            sid="ECRLogInToken",
                            actions=["ecr:GetAuthorizationToken"],
                            effect=Effect.ALLOW,
                            resources=["*"],
                        ),
                        PolicyStatement(
                            sid="ECRReadWrite",
                            actions=[
                                "ecr:CreateRepository",
                                "ecr:BatchGetImage",
                                "ecr:CompleteLayerUpload",
                                "ecr:DescribeImages",
                                "ecr:DescribeRepositories",
                                "ecr:UploadLayerPart",
                                "ecr:ListImages",
                                "ecr:InitiateLayerUpload",
                                "ecr:BatchCheckLayerAvailability",
                                "ecr:PutImage",
                            ],
                            effect=Effect.ALLOW,
                            resources=[
                                # We'll only allow specific repo name(s), but a range in case users
                                # want to experiment with additional custom containers e.g. train,
                                # inference:
                                "arn:aws:ecr:*:*:repository/sm-ocr-*",
                            ],
                        ),
                        PolicyStatement(
                            sid="AccessPreBuiltAWSImages",
                            actions=[
                                "ecr:BatchGetImage",
                                "ecr:GetDownloadUrlForLayer",
                            ],
                            effect=Effect.ALLOW,
                            resources=[
                                "arn:aws:ecr:*:121021644041:repository/*",
                                "arn:aws:ecr:*:763104351884:repository/*",
                                "arn:aws:ecr:*:217643126080:repository/*",
                                "arn:aws:ecr:*:727897471807:repository/*",
                                "arn:aws:ecr:*:626614931356:repository/*",
                                "arn:aws:ecr:*:683313688378:repository/*",
                                "arn:aws:ecr:*:520713654638:repository/*",
                                "arn:aws:ecr:*:462105765813:repository/*",
                            ],
                        ),
                        PolicyStatement(
                            sid="BundleCodeToS3",
                            actions=[
                                "s3:GetObject",
                                "s3:DeleteObject",
                                "s3:PutObject",
                            ],
                            effect=Effect.ALLOW,
                            resources=[
                                # Tightened this up a bit vs the default:
                                # "arn:aws:s3:::sagemaker-*/*"
                                "arn:aws:s3:::sagemaker-*/codebuild-sagemaker-container-*"
                            ],
                        ),
                        # Omit this one because the user should have it already per our guidance,
                        # and if they don't already it's probably best to fail than quietly grant:
                        # PolicyStatement(
                        #     sid="CreateSageMakerDefaultBucketIfMissing",
                        #     actions=["s3:CreateBucket"],
                        #     effect=Effect.ALLOW,
                        #     resources=["arn:aws:s3:::sagemaker*"],
                        # ),
                        # Only required if not explicitly passing a --role (which we will):
                        # PolicyStatement(
                        #     sid="LookUpIAMRoles",
                        #     actions=["iam:GetRole", "iam:ListRoles"],
                        #     effect=Effect.ALLOW,
                        #     resources=["*"],
                        # ),
                        # Only required if building within VPCs (which we won't):
                        # PolicyStatement(
                        #     sid="VPCAccess",
                        #     actions=[
                        #         "ec2:CreateNetworkInterface",
                        #         "ec2:CreateNetworkInterfacePermission",
                        #         "ec2:DescribeDhcpOptions",
                        #         "ec2:DescribeNetworkInterfaces",
                        #         "ec2:DeleteNetworkInterface",
                        #         "ec2:DescribeSubnets",
                        #         "ec2:DescribeSecurityGroups",
                        #         "ec2:DescribeVpcs"
                        #     ],
                        #     effect=Effect.ALLOW,
                        #     resources=["*"],
                        # ),
                    ],
                ),
            },
        )

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
        stack = Stack.of(self)
        Bucket.from_bucket_arn(
            self,
            "SageMakerDefaultBucket",
            f"arn:{stack.partition}:s3:::sagemaker-{stack.region}-{stack.account}",
        ).grant_read_write(self.lambda_role)

        self._pre_lambda = PythonFunction(
            self,
            # Include 'LabelingFunction' in the name so the entities with the
            # AmazonSageMakerGroundTruthExecution policy will automatically have access to call it:
            # https://console.aws.amazon.com/iam/home?#/policies/arn:aws:iam::aws:policy/AmazonSageMakerGroundTruthExecution
            # (Of course this won't work if construct so deeply nested the name is cut off)
            "PreLabelingFunction",
            entry=PRE_LAMBDA_PATH,
            index="main.py",
            handler="handler",
            memory_size=128,
            role=self.lambda_role,
            runtime=LambdaRuntime.PYTHON_3_9,
            timeout=Duration.seconds(5),
        )
        self._post_lambda = PythonFunction(
            self,
            "PostLabelingFunction",
            entry=POST_LAMBDA_PATH,
            index="main.py",
            handler="handler",
            memory_size=128,
            role=self.lambda_role,
            runtime=LambdaRuntime.PYTHON_3_9,
            timeout=Duration.seconds(60),
        )

    @property
    def pre_lambda(self):
        return self._pre_lambda

    @property
    def post_lambda(self):
        return self._post_lambda

    def get_data_science_policy_statements(self) -> List[PolicyStatement]:
        """Generate policy statements required for data scientist to use the annotation infra"""
        return [
            PolicyStatement(
                sid="PassSMImageBuildRole",
                actions=["iam:PassRole"],
                resources=[self.sm_image_build_role.role_arn],
                conditions={
                    "StringLikeIfExists": {
                        "iam:PassedToService": "codebuild.amazonaws.com",
                    },
                },
            ),
            PolicyStatement(
                sid="EditSMStudioCodeBuildProjects",
                actions=[
                    "codebuild:DeleteProject",
                    "codebuild:CreateProject",
                    "codebuild:BatchGetBuilds",
                    "codebuild:StartBuild",
                ],
                effect=Effect.ALLOW,
                resources=["arn:aws:codebuild:*:*:project/sagemaker-studio*"],
            ),
            PolicyStatement(
                sid="InvokeCustomSMGTLambdas",
                actions=[
                    "lambda:InvokeFunction",
                ],
                effect=Effect.ALLOW,
                resources=[
                    self.pre_lambda.function_arn,
                    self.post_lambda.function_arn,
                ],
            ),
        ]
