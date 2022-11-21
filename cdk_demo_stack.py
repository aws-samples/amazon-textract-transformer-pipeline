# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""AWS CDK deployable stack for OCR pipeline sample
"""
# Python Built-Ins:
from typing import List, Optional

# External Dependencies:
from aws_cdk import CfnOutput, CfnParameter, Duration, RemovalPolicy, Stack
from aws_cdk.aws_iam import (
    ManagedPolicy,
    PolicyDocument,
)
import aws_cdk.aws_s3 as s3
import aws_cdk.aws_ssm as ssm
from constructs import Construct

# Local Dependencies:
from annotation import AnnotationInfra
from pipeline import ProcessingPipeline
from pipeline.iam_utils import (
    S3Statement,
    SsmParameterReadStatement,
    StateMachineExecuteStatement,
)


class PipelineDemoStack(Stack):
    """Deployable CDK stack bundling the core OCR pipeline construct with supporting demo resources

    This stack bundles the core ProcessingPipeline construct with the additional resources required
    to deploy and use it for the demo: Such as the project ID SSM parameter, input data bucket,
    SageMaker permissions policy, and the infrastructure for custom annotation UIs in SageMaker
    Ground Truth.

    It also creates several CloudFormation stack outputs to help users find important created
    resources.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        default_project_id: str,
        use_thumbnails: bool,
        enable_sagemaker_autoscaling: bool = False,
        build_sagemaker_ocrs: List[str] = [],
        deploy_sagemaker_ocrs: List[str] = [],
        use_sagemaker_ocr: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Create a PipelineDemoStack

        Parameters
        ----------
        scope :
            As per aws_cdk.Stack
        construct_id :
            As per aws_cdk.Stack
        default_project_id :
            The `ProjectId` is a CFn stack parameter that prefixes created SSM parameters and
            allows SageMaker notebooks to look up the parameters for the deployed stack. If you're
            deploying straight from `cdk deploy`, then the value you specify here will be used. If
            you're `cdk synth`ing a CloudFormation template, then this will be the default value
            for the ProjectId parameter.
        use_thumbnails :
            Set `True` to build the stack with support for visual (page thumbnail image) model
            input features, or `False` to omit the thumbnailing step. Pipelines deployed with
            `use_thumbnails=True` will fail if a thumbnailer endpoint is not set up (see SageMaker
            notebooks). Pipelines deployed with `use_thumbnails=False` cannot fully utilize model
            architectures that use page images for inference (such as LayoutLMv2+, etc).
        enable_sagemaker_autoscaling :
            Set True to enable auto-scale-to-zero on any SageMaker endpoints created by the stack.
            Turning this on should improve cost-efficiency for workloads which are often idle, but
            will introduce cold-start delays to affected stages of the pipeline so may not be ideal
            during development. This setting does not affect endpoints created *outside* the stack
            and later plumbed in to the pipeline (i.e. endpoints deployed from notebooks).
        build_sagemaker_ocrs :
            List of alternative (SageMaker-based) OCR engine names to build container images and
            SageMaker Models for in the deployed stack. By default ([]), none will be included. See
            `CUSTOM_OCR_ENGINES` in pipeline/ocr/sagemaker_ocr.py for supported engines.
        deploy_sagemaker_ocrs :
            List of alternative OCR engine names to deploy SageMaker endpoints for in the stack. Any
            names in here must also be included in `build_sagemaker_ocrs`. Default []: Support
            Amazon Textract OCR only.
        use_sagemaker_ocr :
            Optional alternative OCR engine name to use in the deployed document pipeline. If set
            and not empty, this must also be present in `build_sagemaker_ocrs` and
            `deploy_sagemaker_ocrs`. Default None: Use Amazon Textract for initial document OCR.
        **kwargs :
            As per aws_cdk.Stack
        """
        super().__init__(scope, construct_id, **kwargs)

        # Could consider just directly using the stack ID for this, but then if you were to vend
        # the stack through e.g. AWS Service Catalog you may not have control over setting a nice
        # readable stack ID:
        self.project_id_param = CfnParameter(
            self,
            "ProjectId",
            allowed_pattern=r"[a-zA-Z0-9]+(\-[a-zA-Z0-9]+)*",
            constraint_description="Alphanumeric with internal hyphens allowed",
            default=default_project_id,
            description=(
                "ID to look up this stack's resources from SageMaker notebooks, used in folder "
                "prefixes for SSM parameters."
            ),
            max_length=25,
            min_length=3,
        )

        self.annotation_infra = AnnotationInfra(self, "AnnotationInfra")

        self.input_bucket = s3.Bucket(
            self,
            "PipelineInputBucket",
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess(
                block_public_acls=True,
                block_public_policy=True,
                ignore_public_acls=True,
                restrict_public_buckets=True,
            ),
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            lifecycle_rules=[
                s3.LifecycleRule(enabled=True, expiration=Duration.days(7)),
            ],
            removal_policy=RemovalPolicy.DESTROY,
            cors=[
                # CORS permissions are required for the A2I human review UI to retrieve objects:
                s3.CorsRule(
                    allowed_headers=["*"],
                    allowed_methods=[s3.HttpMethods.GET],
                    allowed_origins=[
                        "https://mturk-console-template-preview-hooks.s3.amazonaws.com",
                    ],
                ),
            ],
        )
        self.pipeline = ProcessingPipeline(
            self,
            "ProcessingPipeline",
            input_bucket=self.input_bucket,
            ssm_param_prefix=f"/{self.project_id_param.value_as_string}/config/",
            use_thumbnails=use_thumbnails,
            enable_sagemaker_autoscaling=enable_sagemaker_autoscaling,
            build_sagemaker_ocrs=build_sagemaker_ocrs,
            deploy_sagemaker_ocrs=deploy_sagemaker_ocrs,
            use_sagemaker_ocr=use_sagemaker_ocr,
        )
        self.data_science_policy = ManagedPolicy(
            self,
            "PipelineDataSciencePolicy",
            document=PolicyDocument(
                statements=[
                    S3Statement(
                        grant_write=True,
                        resources=[self.input_bucket],
                        sid="ReadWritePipelineInputBucket",
                    ),
                    StateMachineExecuteStatement(
                        resources=[self.pipeline.plain_textract_state_machine],
                        sid="RunPlainTextractStateMachine",
                    ),
                ]
                + self.annotation_infra.get_data_science_policy_statements()
                + self.pipeline.config_read_write_statements()
                # In the notebooks we'll use the same execution role for the trained model/endpoint
                # as the notebook itself runs with - so need to grant the role the required perms
                # for reading/writing relevant S3 buckets and publishing to SNS in the pipeline:
                + self.pipeline.sagemaker_model_statements(),
            ),
        )

        # We'd like the stack to push some useful outputs to CloudFormation, and defining them here
        # rather than in the lower-level constructs will keep the constructs flexible/reusable.
        #
        # We override the auto-generated logical IDs to make the names simple to find in console.
        self.data_science_policy_output = CfnOutput(
            self,
            "DataSciencePolicyName",
            description=(
                "Name of the IAM policy with permissions needed for the SageMaker notebooks to "
                "access this stack's resources. Add this policy to your SageMaker execution role."
            ),
            value=self.data_science_policy.managed_policy_name,
        )
        self.data_science_policy_output.override_logical_id("DataSciencePolicyName")
        self.input_bucket_name_output = CfnOutput(
            self,
            "InputBucketName",
            description="Name of the S3 bucket to which input documents should be uploaded",
            value=self.pipeline.input_bucket.bucket_name,
        )
        self.input_bucket_name_output.override_logical_id("InputBucketName")
        self.pipeline_statemachine_output = CfnOutput(
            self,
            "PipelineStateMachine",
            description="ARN of the State Machine for the end-to-end OCR pipeline",
            value=self.pipeline.state_machine.state_machine_arn,
        )
        self.pipeline_statemachine_output.override_logical_id("PipelineStateMachine")
        self.textract_statemachine_output = CfnOutput(
            self,
            "PlainTextractStateMachine",
            description="ARN of the State Machine for *only* running Textract (no enrichments)",
            value=self.pipeline.plain_textract_state_machine.state_machine_arn,
        )
        self.textract_statemachine_output.override_logical_id("PlainTextractStateMachine")
        self.model_param_output = CfnOutput(
            self,
            "SageMakerEndpointParamName",
            description="SSM parameter to configure the pipeline's SageMaker endpoint name",
            value=self.pipeline.sagemaker_endpoint_param.parameter_name,
        )
        self.model_param_output.override_logical_id("SageMakerEndpointParamName")
        self.thumbnail_param_output = CfnOutput(
            self,
            "ThumbnailEndpointParamName",
            description=(
                "SSM parameter to configure the pipeline's Thumbnail generation endpoint name"
            ),
            value="undefined"
            if self.pipeline.thumbnail_endpoint_param is None
            else self.pipeline.thumbnail_endpoint_param.parameter_name,
        )
        self.thumbnail_param_output.override_logical_id("ThumbnailEndpointParamName")
        self.entity_config_param_output = CfnOutput(
            self,
            "EntityConfigParamName",
            description=(
                "JSON configuration describing the field types to be extracted by the pipeline"
            ),
            value=self.pipeline.entity_config_param.parameter_name,
        )
        self.entity_config_param_output.override_logical_id("EntityConfigParamName")
        self.a2i_role_arn_output = CfnOutput(
            self,
            "A2IHumanReviewExecutionRoleArn",
            description="ARN of the execution Role to use for Amazon A2I human review workflows",
            value=self.pipeline.review_a2i_role.role_arn,
        )
        self.a2i_role_arn_output.override_logical_id("A2IHumanReviewExecutionRoleArn")
        self.workflow_param_output = CfnOutput(
            self,
            "A2IHumanReviewFlowParamName",
            description="SSM parameter to configure the pipeline's A2I review workflow ARN",
            value=self.pipeline.review_workflow_param.parameter_name,
        )
        self.workflow_param_output.override_logical_id("A2IHumanReviewFlowParamName")
        self.reviews_bucket_name_output = CfnOutput(
            self,
            "A2IHumanReviewBucketName",
            description="Name of the S3 bucket to which A2I reviews should be stored",
            value=self.pipeline.human_reviews_bucket.bucket_name,
        )
        self.reviews_bucket_name_output.override_logical_id("A2IHumanReviewBucketName")

        # While these CloudFormation outputs are nice for the CFn console, we'd also like to be
        # able to automatically look up project resources from SageMaker notebooks. To support
        # this, we'll create additional SSM params used just to *retrieve* static attributes of the
        # stack - rather than configuration points like the ProcessingPipeline construct's params.
        static_param_prefix = f"/{self.project_id_param.value_as_string}/static"
        self.sm_image_build_role_ssm_param = ssm.StringParameter(
            self,
            "SMImageBuildRoleSSMParam",
            string_value=self.annotation_infra.sm_image_build_role.role_name,
            description=(
                "Name of the CodeBuild execution role to use in SMStudio Image Build commands"
            ),
            parameter_name=f"{static_param_prefix}/SMDockerBuildRole",
            simple_name=False,
        )
        self.preproc_image_param = ssm.StringParameter(
            self,
            "PreprocImageSSMParam",
            description="URI of the thumbnail generator container image pre-created by the stack",
            parameter_name=f"{static_param_prefix}/PreprocImageURI",
            simple_name=False,
            string_value=self.pipeline.preproc_image.image_uri,
        )
        self.input_bucket_ssm_param = ssm.StringParameter(
            self,
            "InputBucketNameSSMParam",
            string_value=self.input_bucket.bucket_name,
            description="Name of the S3 bucket to which input documents should be uploaded",
            parameter_name=f"{static_param_prefix}/InputBucket",
            simple_name=False,
        )
        self.reviews_bucket_ssm_param = ssm.StringParameter(
            self,
            "ReviewsBucketNameSSMParam",
            string_value=self.pipeline.human_reviews_bucket.bucket_name,
            description="Name of the S3 bucket to which human reviews should be stored",
            parameter_name=f"{static_param_prefix}/ReviewsBucket",
            simple_name=False,
        )
        self.pipeline_statemachine_ssm_param = ssm.StringParameter(
            self,
            "PipelineStateMachineSSMParam",
            string_value=self.pipeline.state_machine.state_machine_arn,
            description="ARN of the State Machine for the end-to-end OCR pipeline",
            parameter_name=f"{static_param_prefix}/PipelineStateMachine",
            simple_name=False,
        )
        self.textract_statemachine_ssm_param = ssm.StringParameter(
            self,
            "PlainTextractStateMachineSSMParam",
            string_value=self.pipeline.plain_textract_state_machine.state_machine_arn,
            description="ARN of the State Machine for *only* running Textract (no enrichments)",
            parameter_name=(
                f"/{self.project_id_param.value_as_string}/static/PlainTextractStateMachine"
            ),
            simple_name=False,
        )
        self.enrichment_results_bucket_ssm_param = ssm.StringParameter(
            self,
            "EnrichmentModelResultsBucketSSMParam",
            description=(
                "Name of the S3 bucket to which SageMaker (async) model results should be stored"
            ),
            parameter_name=f"{static_param_prefix}/ModelResultsBucket",
            simple_name=False,
            string_value=self.pipeline.enriched_results_bucket.bucket_name,
        )
        self.thumbnails_callback_topic_ssm_param = ssm.StringParameter(
            self,
            "ThumbnailsCallbackTopicSSMParam",
            description="ARN of the SNS Topic to use for thumbnail images generation callback",
            parameter_name=f"{static_param_prefix}/ThumbnailsCallbackTopicArn",
            simple_name=False,
            string_value=(
                self.pipeline.thumbnail_sns_topic.topic_arn
                if self.pipeline.thumbnail_sns_topic
                else "undefined"
            ),
        )
        self.enrichment_callback_topic_ssm_param = ssm.StringParameter(
            self,
            "EnrichmentModelCallbackTopicSSMParam",
            description="ARN of the SNS Topic to use for callback in SageMaker Async Inference",
            parameter_name=f"{static_param_prefix}/ModelCallbackTopicArn",
            simple_name=False,
            string_value=(
                self.pipeline.sagemaker_sns_topic.topic_arn
                if self.pipeline.sagemaker_sns_topic
                else "undefined"
            ),
        )
        self.a2i_role_arn_param = ssm.StringParameter(
            self,
            "A2IExecutionRoleArnParam",
            string_value=self.pipeline.review_a2i_role.role_arn,
            description="ARN of the execution role which A2I human review workflows should use",
            parameter_name=f"{static_param_prefix}/A2IExecutionRoleArn",
            simple_name=False,
        )

        self.data_science_policy.add_statements(
            SsmParameterReadStatement(
                resources=[
                    self.sm_image_build_role_ssm_param,
                    self.input_bucket_ssm_param,
                    self.reviews_bucket_ssm_param,
                    self.pipeline_statemachine_ssm_param,
                    self.preproc_image_param,
                    self.textract_statemachine_ssm_param,
                    self.thumbnails_callback_topic_ssm_param,
                    self.enrichment_callback_topic_ssm_param,
                    self.enrichment_results_bucket_ssm_param,
                    self.a2i_role_arn_param,
                ],
                sid="ReadStaticPipelineParams",
            ),
        )
