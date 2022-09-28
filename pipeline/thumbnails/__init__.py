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
import aws_cdk.aws_iam as iam
import aws_cdk.aws_s3 as s3
from aws_cdk.aws_sns import ITopic, Topic
import aws_cdk.aws_ssm as ssm
import aws_cdk.aws_stepfunctions as sfn
from constructs import Construct

# Local Dependencies:
from ..shared import abs_path
from ..shared.sagemaker import (
    SageMakerAsyncInferenceConfig,
    SageMakerCallerFunction,
    SageMakerSSMStep,
    SageMakerDLCBasedImage,
    SageMakerCustomizedDLCModel,
    SageMakerModelDeployment,
)


class ThumbnailGeneratorDeployment(Construct):
    """Construct to deploy the thumbnailer endpoint along with the stack (rather than via notebook)"""

    def __init__(
        self,
        scope: Construct,
        id: str,
        thumbnailer_name: str,
        s3_output_bucket: s3.Bucket,
        s3_output_prefix: str = "",
        sns_success_topic: Optional[ITopic] = None,
        sns_error_topic: Optional[ITopic] = None,
    ):
        """Create a ThumbnailGeneratorDeployment

        Parameters
        ----------
        scope :
            As per CDK Construct parent.
        id :
            As per CDK Construct parent.
        thumbnailer_name :
            Name to use for SageMaker Model/Endpoint/EndpointConfig
        s3_output_bucket :
            Bucket in which thumbnail outputs should be stored (for SM Async Inference)
        s3_output_prefix :
            Prefix under `s3_output_bucket` in which thumbnail outputs should be stored (for SM
            Async Inference)
        sns_success_topic :
            SNS Topic to notify on successful inference (for SM Async Inference)
        sns_error_topic :
            SNS Topic to notify on failed inference (for SM Async Inference)
        """
        super().__init__(scope, id)

        self.image = SageMakerDLCBasedImage(
            self,
            "Image",
            directory=abs_path("../../notebooks/custom-containers/preproc", __file__),
            file="Dockerfile",
            ecr_repo="sm-ocr-preproc",
            ecr_tag="pytorch-1.10-inf-cpu",
            framework="pytorch",
            use_gpu=False,
            image_scope="inference",
            py_version="py38",
            version="1.10",
        )

        thumbnailer_role = iam.Role(
            self,
            "ThumbnailerRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="Role for CDK-created SageMaker endpoints in OCR Pipeline",
        )
        # S3 and SNS permissions will be configured automatically by the Model/Deployment constructs

        self.model = SageMakerCustomizedDLCModel(
            self,
            "Model",
            model_name=thumbnailer_name,
            image=self.image,
            execution_role=thumbnailer_role,
            entry_point="inference.py",
            source_dir=abs_path("../../notebooks/preproc", __file__),
        )

        self.deployment = SageMakerModelDeployment(
            self,
            "Deployment",
            endpoint_name=thumbnailer_name,
            model=self.model,
            instance_type="ml.m5.xlarge",
            initial_instance_count=1,
            async_inference_config=SageMakerAsyncInferenceConfig(
                s3_output_bucket=s3_output_bucket,
                s3_output_prefix=s3_output_prefix,
                sns_success_topic=sns_success_topic,
                sns_error_topic=sns_error_topic,
                max_concurrent_invocations_per_instance=2,
            ),
        )

    @property
    def endpoint_name(self) -> str:
        return self.deployment.endpoint.endpoint_name


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
        lambda_role: iam.Role,
        ssm_param_prefix: Union[Token, str],
        thumbnails_bucket: s3.Bucket,
        thumbnails_prefix: str = "",
        auto_deploy_thumbnailer: bool = False,
        shared_sagemaker_caller_lambda: Optional[SageMakerCallerFunction] = None,
        **kwargs,
    ):
        """Create a GenerateThumbnailsStep

        Parameters
        ----------
        scope :
            As per CDK Construct parent.
        id :
            As per CDK Construct parent.
        lambda_role :
            Shared execution role to use for Step Functions integration Lambda.
        ssm_param_prefix :
            Prefix to be applied to generated SSM pipeline configuration parameter names (including
            the parameter to configure SageMaker endpoint name for thumbnail generation).
        thumbnails_bucket :
            Bucket into which thumbnail results should be saved. This is only used when
            `deploy_auto_thumbnailer=True`: Otherwise will be configured at the point a user creates
            the endpoint from notebooks.
        thumbnails_prefix :
            Prefix under `thumbnails_bucket` where thumbnail results should be saved. This is only
            used when `deploy_auto_thumbnailer=True`: Otherwise will be configured at the point a
            user creates the endpoint from notebooks.
        auto_deploy_thumbnailer :
            Set `True` to also automatically deploy the SageMaker endpoint for thumbnail generation.
            Default `False` expects this to be configured later from the walkthrough notebooks.
        shared_sagemaker_caller_lambda :
            Optional pre-existing SageMaker caller Lambda function, to share this between multiple
            SageMakerSSMSteps in the app if required.
        """
        super().__init__(scope, id, **kwargs)

        async_callback_topic = Topic(self, f"SageMakerAsync-{id}")

        if auto_deploy_thumbnailer:
            self.deployment = ThumbnailGeneratorDeployment(
                self,
                "Thumbnailer",
                thumbnailer_name="ocr-thumbnail-cdk-8",
                s3_output_bucket=thumbnails_bucket,
                s3_output_prefix=thumbnails_prefix,
                sns_success_topic=async_callback_topic,
                sns_error_topic=async_callback_topic,
            )
        else:
            self.deployment = None

        self.endpoint_param = ssm.StringParameter(
            self,
            "ThumbnailGeneratorEndpointParam",
            description="Name of the SageMaker Endpoint to call for generating page thumbnails",
            parameter_name=f"{ssm_param_prefix}ThumbnailEndpointName",
            simple_name=False,
            string_value=self.deployment.endpoint_name if self.deployment else "undefined",
        )
        lambda_role.add_to_policy(
            iam.PolicyStatement(
                sid="ReadThumbnailEndpointParam",
                actions=["ssm:GetParameter"],
                effect=iam.Effect.ALLOW,
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

    def sagemaker_sns_statements(
        self,
        sid_prefix: Union[str, None] = "",
    ) -> List[iam.PolicyStatement]:
        """Create PolicyStatements to grant SageMaker permission to use the SNS callback topic

        Arguments
        ---------
        sid_prefix : str | None
            Prefix to add to generated statement IDs for uniqueness, or "", or None to suppress
            SIDs.
        """
        return self.sfn_task.sagemaker_sns_statements(sid_prefix=sid_prefix)
