# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CDK for page thumbnail image generation stage of the OCR pipeline

Some multi-modal document analysis models can consume images of the pages as input features. To do
this, we need clean images in a standard size and format to feed in to the model. This component
processes raw document inputs into thumbnail image bundles in (near-) real time via a SageMaker
endpoint.
"""
# Python Built-Ins:
from typing import List, Optional, Union

# External Dependencies:
from aws_cdk import Duration, Token
from aws_cdk.aws_iam import Effect, PolicyStatement, Role
import aws_cdk.aws_ssm as ssm
import aws_cdk.aws_stepfunctions as sfn
from constructs import Construct

# Local Dependencies:
from ..shared.sagemaker import SageMakerCallerFunction, SageMakerSSMStep


class GenerateThumbnailsStep(Construct):
    """CDK construct for an OCR pipeline step to generate page thumbnails using SageMaker

    This construct's `.sfn_task` expects inputs with $.Input.Bucket and $.Input.Key properties
    specifying the location of the raw input document, and will return an object with Bucket and
    Key pointing to the consolidated page thumbnail images (.npz archive) object.

    This step is implemented via AWS Lambda (rather than direct Step Function service call) to
    support looking up the configured SageMaker endpoint name from SSM within the same SFn step.
    """

    def __init__(
        self,
        scope: Construct,
        id: str,
        lambda_role: Role,
        ssm_param_prefix: Union[Token, str],
        shared_sagemaker_caller_lambda: Optional[SageMakerCallerFunction] = None,
        **kwargs,
    ):
        super().__init__(scope, id, **kwargs)

        self.endpoint_param = ssm.StringParameter(
            self,
            "ThumbnailGeneratorEndpointParam",
            description="Name of the SageMaker Endpoint to call for generating page thumbnails",
            parameter_name=f"{ssm_param_prefix}ThumbnailEndpointName",
            simple_name=False,
            string_value="undefined",
        )
        lambda_role.add_to_policy(
            PolicyStatement(
                sid="ReadThumbnailEndpointParam",
                actions=["ssm:GetParameter"],
                effect=Effect.ALLOW,
                resources=[self.endpoint_param.parameter_arn],
            )
        )

        self.sfn_task = SageMakerSSMStep(
            self,
            "GenerateThumbnails",
            lambda_role=lambda_role,
            support_async_endpoints=True,
            comment="Post-Process the Textract result with Amazon SageMaker",
            lambda_function=shared_sagemaker_caller_lambda,
            payload=sfn.TaskInput.from_object(
                {
                    "EndpointNameParam": self.endpoint_param.parameter_name,
                    "Accept": "application/x-npz",
                    "InputLocation": {
                        "Bucket": sfn.JsonPath.string_at("$.Input.Bucket"),
                        "Key": sfn.JsonPath.string_at("$.Input.Key"),
                    },
                    "TaskToken": sfn.JsonPath.task_token,
                }
            ),
            # Actual invocation should be fairly quick - but of course the request may get queued
            # or async endpoint may need to scale up from 0 instances... So give a bit of room:
            timeout=Duration.minutes(30),
        )

    def sagemaker_sns_statements(self, sid_prefix: Union[str, None] = "") -> List[PolicyStatement]:
        """Create PolicyStatements to grant SageMaker permission to use the SNS callback topic

        Arguments
        ---------
        sid_prefix : str | None
            Prefix to add to generated statement IDs for uniqueness, or "", or None to suppress
            SIDs.
        """
        return self.sfn_task.sagemaker_sns_statements(sid_prefix=sid_prefix)
