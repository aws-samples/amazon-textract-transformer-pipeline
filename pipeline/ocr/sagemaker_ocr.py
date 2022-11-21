# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CDK for alternative OCR (Non-Amazon Textract) stage of the document processing pipeline
"""
# Python Built-Ins:
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

# External Dependencies:
from aws_cdk import CfnTag, Duration, Token
import aws_cdk.aws_iam as iam
from aws_cdk.aws_s3 import Bucket
import aws_cdk.aws_sns as sns
import aws_cdk.aws_ssm as ssm
import aws_cdk.aws_stepfunctions as sfn
from constructs import Construct

# Local Dependencies:
from ..shared import abs_path
from ..shared.sagemaker import (
    EndpointAutoscaler,
    SageMakerAsyncInferenceConfig,
    SageMakerAutoscalingRole,
    SageMakerCustomizedDLCModel,
    SageMakerCallerFunction,
    SageMakerDLCBasedImage,
    SageMakerDLCSpec,
    SageMakerEndpointExecutionRole,
    SageMakerModelDeployment,
    SageMakerSSMStep,
)


@dataclass
class CustomOCREngineSpec:
    """Data class for configuring a custom OCR engine integration

    Parameters
    ----------
    base_dlc :
        SageMaker Deep Learning Container to use as a base for customized container image
    build_folder :
        Build context folder for customized container image
    ecr_repo :
        ECR Repository name where the customized container image should be stored
    ecr_tag :
        Tag for the customized ECR container image
    instance_type :
        SageMaker instance type to deploy for the endpoint (e.g. `ml.m5.xlarge`)
    entry_point :
        Path (relative to `source_dir` to the inference entry-point script)
    source_dir :
        Folder of inference scripts to bundle to for the endpoint
    dockerfile_relpath :
        Path relative to `build_folder` to the container Dockerfile. Default "Dockerfile"
    build_args :
        { name: value } dictionary of arguments for container build
    environment :
        { name: value } dictionary of environment variables to set on the model/endpoint
    max_async_invocations_per_instance :
        Limit of concurrent requests each instance on the endpoint should process
    """

    base_dlc: SageMakerDLCSpec
    build_folder: str
    ecr_repo: str
    ecr_tag: str
    instance_type: str
    entry_point: str
    source_dir: str
    dockerfile_relpath: str = "Dockerfile"
    build_args: Optional[Dict[str, str]] = None
    environment: Optional[Dict[str, str]] = None
    max_async_invocations_per_instance: int = 2


# Define your additional custom OCR integrations here:
CUSTOM_OCR_ENGINES: Dict[str, CustomOCREngineSpec] = {
    "tesseract": CustomOCREngineSpec(
        base_dlc=SageMakerDLCSpec(
            framework="pytorch",
            use_gpu=False,
            image_scope="inference",
            py_version="py38",
            version="1.10",
        ),
        build_args={"INCLUDE_OCR_TESSERACT": "true"},
        build_folder=abs_path("../../notebooks/custom-containers/preproc", __file__),
        ecr_repo="sm-ocr-engines",
        ecr_tag="ocr-tesseract",
        entry_point="ocr.py",
        source_dir=abs_path("../../notebooks/preproc", __file__),
        environment={
            "OCR_ENGINE": "tesseract",
            "OCR_DEFAULT_LANGUAGES": "eng,tha",
            "OCR_DEFAULT_DPI": "300",
        },
        instance_type="ml.m5.xlarge",
    )
}


class SageMakerOCRStep(Construct):
    """CDK construct for an alternative (non-Textract) OCR step in SFn-based document pipeline

    This construct's `.sfn_task` expects inputs with $.Input.Bucket and $.Input.Key properties
    specifying the location of the raw input document, and will return an object with Bucket and
    Key pointing to the consolidated Textract JSON object.
    """

    def __init__(
        self,
        scope: Construct,
        id: str,
        lambda_role: iam.Role,
        ssm_param_prefix: Union[Token, str],
        input_bucket: Bucket,
        ocr_results_bucket: Bucket,
        input_prefix: Optional[str] = None,
        ocr_results_prefix: str = "",
        build_engine_names: List[str] = ["tesseract"],
        deploy_engine_names: List[str] = ["tesseract"],
        use_engine_name: Optional[str] = "tesseract",
        enable_autoscaling: bool = False,
        shared_sagemaker_caller_lambda: Optional[SageMakerCallerFunction] = None,
        timeout_including_queue: Duration = Duration.minutes(30),
        **kwargs,
    ):
        """Create a SageMakerOCRStep

        Parameters
        ----------
        scope :
            CDK construct scope
        id :
            CDK construct ID
        lambda_role :
            IAM Role that the Amazon Textract-invoking Lambda function will run with
        ssm_param_prefix :
            Prefix to be applied to generated SSM pipeline configuration parameter names (including
            the parameter to configure SageMaker endpoint name for thumbnail generation).
        input_bucket :
            Bucket from which input documents will be fetched. If auto-deployment of a thumbnailer
            endpoint is enabled, the model execution role will be granted access to this bucket
            (limited to `input_prefix`).
        ocr_results_bucket :
            (Pre-existing) S3 bucket where Textract result files should be stored
        input_prefix :
            Prefix under `input_bucket` from which input documents will be fetched. Used to
            configure SageMaker model execution role permissions when auto-deployment of thumbnailer
            endpoint is enabled.
        ocr_results_prefix :
            Prefix under which Textract result files should be stored in S3 (under this prefix,
            the original input document keys will be mapped).
        build_engine_names :
            List of custom OCR engine names as per `CUSTOM_OCR_ENGINES`, to build container images
            and SageMaker Models for.
        deploy_engine_names :
            List of custom OCR engine names to deploy SageMaker endpoints for. Any names in here
            must also be included in `build_engine_names`.
        use_engine_name :
            Name of which custom OCR engine to reference in the Step Functions pipeline step.
        enable_autoscaling :
            Set True to enable auto-scaling on the endpoint to optimize resource usage (recommended
            for production use), or False to disable it and avoid cold-starts (good for development)
        shared_sagemaker_caller_lambda :
            Optional pre-existing SageMaker caller Lambda function, to share this between multiple
            SageMakerSSMSteps in the app if required.
        timeout_including_queue :
            Timeout for the end-to-end OCR step (including concurrency management / queuing) to be
            considered as failed.
        """
        super().__init__(scope, id, **kwargs)

        # Validate parameters:
        for name in build_engine_names:
            if name not in CUSTOM_OCR_ENGINES:
                raise ValueError(f"OCR engine name '{name}' not found in CUSTOM_OCR_ENGINES config")
        for name in deploy_engine_names:
            if name not in build_engine_names:
                raise ValueError(
                    f"deploy_engine_names must all be present in build_engine_names. Got: '{name}'"
                )
        if use_engine_name == "":
            use_engine_name = None
        if use_engine_name and (use_engine_name not in deploy_engine_names):
            raise ValueError(
                f"use_engine_name '{use_engine_name}' must be in deploy list {deploy_engine_names}"
            )

        # Create shared resources:
        async_callback_topic = sns.Topic(self, f"SageMakerAsync-{id}")
        sagemaker_role = SageMakerEndpointExecutionRole(
            self,
            "SMOCRRole",
            description="Role for open source-based SageMaker OCR models/endpoints",
        )
        input_bucket.grant_read(sagemaker_role, input_prefix)

        self.images: Dict[SageMakerDLCBasedImage] = {}
        self.models: Dict[str, SageMakerCustomizedDLCModel] = {}
        self.deployments: Dict[str, SageMakerModelDeployment] = {}
        self.autoscalers: Dict[str, Optional[EndpointAutoscaler]] = {}
        if enable_autoscaling:
            autoscaling_role = SageMakerAutoscalingRole(self, "AutoScalingRole")
        else:
            autoscaling_role = None

        # Build images & SM Models for requested engines:
        for name in build_engine_names:
            spec = CUSTOM_OCR_ENGINES[name]
            self.images[name] = SageMakerDLCBasedImage(
                self,
                f"Image-{name}",
                directory=spec.build_folder,
                build_args=spec.build_args,
                file=spec.dockerfile_relpath,
                ecr_repo=spec.ecr_repo,
                ecr_tag=spec.ecr_tag,
                base_image_spec=SageMakerDLCSpec(
                    framework=spec.base_dlc.framework,
                    use_gpu=spec.base_dlc.use_gpu,
                    image_scope=spec.base_dlc.image_scope,
                    py_version=spec.base_dlc.py_version,
                    version=spec.base_dlc.version,
                ),
            )
            self.models[name] = SageMakerCustomizedDLCModel(
                self,
                f"Model-{name}",
                image=self.images[name],
                execution_role=sagemaker_role,
                entry_point=spec.entry_point,
                source_dir=spec.source_dir,
                environment=spec.environment,
                tags=[CfnTag(key="OCREngineName", value=name)],
            )

        # Deploy subset of requested engines to SM endpoints:
        for name in deploy_engine_names:
            spec = CUSTOM_OCR_ENGINES[name]
            self.deployments[name] = SageMakerModelDeployment(
                self,
                f"Deployment-{name}",
                model=self.models[name],
                instance_type="ml.m5.xlarge",
                initial_instance_count=1,
                async_inference_config=SageMakerAsyncInferenceConfig(
                    s3_output_bucket=ocr_results_bucket,
                    s3_output_prefix=ocr_results_prefix,
                    sns_success_topic=async_callback_topic,
                    sns_error_topic=async_callback_topic,
                    max_concurrent_invocations_per_instance=spec.max_async_invocations_per_instance,
                ),
                tags=[CfnTag(key="OCREngineName", value=name)],
            )
            if enable_autoscaling:
                autoscaler = EndpointAutoscaler(
                    self,
                    f"Autoscaling-{name}",
                    max_capacity=4,
                    min_capacity=0,
                    endpoint_name=self.deployments[name].endpoint_name,
                    role=autoscaling_role,
                )
                autoscaler.scale_async_endpoint_simple()
                self.autoscalers[name] = autoscaler

        # Select ONE of the deployed engines to use in the pipeline:
        self.use_engine_name = use_engine_name
        self.endpoint_param = ssm.StringParameter(
            self,
            "SageMakerOCREndpointParam",
            description="Name of the SageMaker Endpoint to call for custom OCR",
            parameter_name=f"{ssm_param_prefix}SMOCREndpointName",
            simple_name=False,
            string_value=(
                self.deployments[use_engine_name].endpoint_name if use_engine_name else "undefined"
            ),
        )
        lambda_role.add_to_policy(
            iam.PolicyStatement(
                sid="ReadSMOCREndpointParam",
                actions=["ssm:GetParameter"],
                effect=iam.Effect.ALLOW,
                resources=[self.endpoint_param.parameter_arn],
            )
        )

        self.sfn_task = SageMakerSSMStep(
            self,
            "SageMakerAltOCR",
            lambda_role=lambda_role,
            support_async_endpoints=True,
            comment="OCR the document with an open-source-based engine on Amazon SageMaker",
            async_notify_topic=async_callback_topic,
            lambda_function=shared_sagemaker_caller_lambda,
            payload=sfn.TaskInput.from_object(
                {
                    "EndpointNameParam": self.endpoint_param.parameter_name,
                    "Accept": "application/json",
                    "InputLocation": {
                        "Bucket": sfn.JsonPath.string_at("$.Input.Bucket"),
                        "Key": sfn.JsonPath.string_at("$.Input.Key"),
                    },
                    "TaskToken": sfn.JsonPath.task_token,
                }
            ),
            # Actual invocation should be fairly quick - but of course the request may get queued
            # or async endpoint may need to scale up from 0 instances... So give a bit of room:
            timeout=timeout_including_queue,
        )

    def sagemaker_sns_statements(
        self,
        sid_prefix: Union[str, None] = "",
    ) -> List[iam.PolicyStatement]:
        """Create PolicyStatements to grant SageMaker permission to use the SNS callback topic

        Parameters
        ----------
        sid_prefix : str | None
            Prefix to add to generated statement IDs for uniqueness, or "", or None to suppress
            SIDs.
        """
        return self.sfn_task.sagemaker_sns_statements(sid_prefix=sid_prefix)
