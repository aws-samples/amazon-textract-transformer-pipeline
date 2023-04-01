# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CDK Construct for an OCR pipeline with field post-processing and human review
"""
# Python Built-Ins:
from typing import List, Optional, Union

# External Dependencies:
from aws_cdk import Duration, Fn, RemovalPolicy, Token
from aws_cdk.aws_iam import (
    Effect,
    ManagedPolicy,
    PolicyStatement,
    Role,
    ServicePrincipal,
)
from aws_cdk.aws_lambda import Runtime as LambdaRuntime
from aws_cdk.aws_lambda_python_alpha import PythonFunction
import aws_cdk.aws_s3 as s3
import aws_cdk.aws_s3_notifications as s3n
import aws_cdk.aws_stepfunctions as sfn
import aws_cdk.aws_sns as sns
import aws_cdk.aws_ssm as ssm
from constructs import Construct

# Local Dependencies:
from .enrichment import SageMakerEnrichmentStep
from .iam_utils import (
    S3Statement,
    SsmParameterReadStatement,
    SsmParameterWriteStatement,
)
from .ocr import OCRStep
from .postprocessing import LambdaPostprocStep
from .review import A2IReviewStep
from .shared import abs_path
from .shared.sagemaker import (
    get_sagemaker_default_bucket,
    SageMakerCallerFunction,
    SageMakerDLCBasedImage,
    SageMakerDLCSpec,
)
from .thumbnails import GenerateThumbnailsStep

S3_TRIGGER_LAMBDA_PATH = abs_path("fn-trigger", __file__)


class ProcessingPipeline(Construct):
    """CDK construct for an OCR pipeline with field post-processing and human review

    This is the main top-level construct for the sample solution, implementing an AWS Step
    Functions-based pipeline orchestrating the different stages of the pipeline.
    """

    def __init__(
        self,
        scope: Construct,
        id: str,
        input_bucket: s3.Bucket,
        ssm_param_prefix: Union[Token, str],
        use_thumbnails: bool = True,
        enable_sagemaker_autoscaling: bool = False,
        build_sagemaker_ocrs: List[str] = [],
        deploy_sagemaker_ocrs: List[str] = [],
        use_sagemaker_ocr: Optional[str] = None,
        **kwargs,
    ):
        """Create a ProcessingPipeline

        Arguments
        ---------
        scope :
            CDK construct scope
        id :
            CDK construct ID
        input_bucket :
            The raw input bucket, to which this pipeline will attach and listen for documents being
            uploaded.
        ssm_param_prefix :
            A prefix to apply to generated AWS SSM Parameter Store configuration parameter names,
            to help keep the account tidy. Should begin and end with a forward slash.
        use_thumbnails :
            When `True`, the pipeline is configured to call a page thumbnail image generation
            endpoint and pass these resized images through to the SageMaker document understanding
            / enrichment model. Set `False` to skip deploying these features. Pipelines deployed
            with `use_thumbnails=True` will fail if a thumbnailer endpoint is not set up (see
            SageMaker notebooks). Pipelines deployed with `use_thumbnails=False` cannot fully
            utilize model architectures that use page images for inference (such as LayoutLMv2+).
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
        **kwargs : Any
            Passed through to parent Construct
        """
        super().__init__(scope, id, **kwargs)

        self.input_bucket = input_bucket
        self.textract_results_bucket = s3.Bucket(
            self,
            "TextractResultsBucket",
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess(
                block_public_acls=True,
                block_public_policy=True,
                ignore_public_acls=True,
                restrict_public_buckets=True,
            ),
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            lifecycle_rules=[s3.LifecycleRule(enabled=True, expiration=Duration.days(7))],
            removal_policy=RemovalPolicy.DESTROY,
        )
        self.enriched_results_bucket = s3.Bucket(
            self,
            "EnrichedResultsBucket",
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess(
                block_public_acls=True,
                block_public_policy=True,
                ignore_public_acls=True,
                restrict_public_buckets=True,
            ),
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            lifecycle_rules=[s3.LifecycleRule(enabled=True, expiration=Duration.days(7))],
            removal_policy=RemovalPolicy.DESTROY,
        )
        self.human_reviews_bucket = s3.Bucket(
            self,
            "HumanReviewsBucket",
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess(
                block_public_acls=True,
                block_public_policy=True,
                ignore_public_acls=True,
                restrict_public_buckets=True,
            ),
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            lifecycle_rules=[s3.LifecycleRule(enabled=True, expiration=Duration.days(7))],
            removal_policy=RemovalPolicy.DESTROY,
        )

        self.shared_lambda_role = Role(
            self,
            "SharedLambdaRole",
            assumed_by=ServicePrincipal("lambda.amazonaws.com"),
            description="Shared execution role for OCR pipeline Lambda functions",
            managed_policies=[
                ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                ),
                ManagedPolicy.from_aws_managed_policy_name("AWSXRayDaemonWriteAccess"),
            ],
        )
        self.input_bucket.grant_read(self.shared_lambda_role)
        self.textract_results_bucket.grant_read_write(self.shared_lambda_role)
        self.enriched_results_bucket.grant_read_write(self.shared_lambda_role)
        self.human_reviews_bucket.grant_read_write(self.shared_lambda_role)
        get_sagemaker_default_bucket(self).grant_read_write(self.shared_lambda_role)

        # The thumbnail generation and enrichment model steps can share a SageMaker integration
        # Lambda function:
        self.shared_sagemaker_lambda = SageMakerCallerFunction(
            self,
            "CallSageMaker",
            support_async_endpoints=True,
            role=self.shared_lambda_role,
            description="Lambda function to invoke SSM-parameterized SageMaker endpoints from SFn",
        )

        self.preproc_image = SageMakerDLCBasedImage(
            self,
            "PreprocessingImage",
            directory=abs_path("../notebooks/custom-containers/preproc", __file__),
            file="Dockerfile",
            ecr_repo="sm-ocr-preprocs",
            ecr_tag="pytorch-1.10-inf-cpu",
            base_image_spec=SageMakerDLCSpec(
                framework="pytorch",
                use_gpu=False,
                image_scope="inference",
                py_version="py38",
                version="1.10",
            ),
        )

        self.ocr_step = OCRStep(
            self,
            "OCRStep",
            lambda_role=self.shared_lambda_role,
            ssm_param_prefix=ssm_param_prefix,
            input_bucket=self.input_bucket,
            output_bucket=self.textract_results_bucket,
            output_prefix="textract",
            build_sagemaker_ocrs=build_sagemaker_ocrs,
            deploy_sagemaker_ocrs=deploy_sagemaker_ocrs,
            use_sagemaker_ocr=use_sagemaker_ocr,
            enable_sagemaker_autoscaling=enable_sagemaker_autoscaling,
            shared_sagemaker_caller_lambda=self.shared_sagemaker_lambda,
        )
        if use_thumbnails:
            self.thumbnails_step = GenerateThumbnailsStep(
                self,
                "ThumbnailStep",
                lambda_role=self.shared_lambda_role,
                ssm_param_prefix=ssm_param_prefix,
                shared_sagemaker_caller_lambda=self.shared_sagemaker_lambda,
                input_bucket=self.input_bucket,
                thumbnails_bucket=self.enriched_results_bucket,
                thumbnails_prefix="preproc",
                container_image=self.preproc_image,
                auto_deploy_thumbnailer=True,
                enable_autoscaling=enable_sagemaker_autoscaling,
            )
            ocr_preproc_states = [self.ocr_step.sfn_task, self.thumbnails_step.sfn_task]
        else:
            self.thumbnails_step = None
            ocr_preproc_states = [self.ocr_step.sfn_task]

        # You could optimize out this Parallel if you're always going to run only the OCR step at
        # this point, but it helps for an agnostic solution because it can be tricky to
        # simultaneously select a subset of the state output and map it to a subkey of the state
        # inside the OCR step.
        ocr_preproc = sfn.Parallel(
            self,
            "OCRAndPreProcessing",
            comment="Run OCR and (if set up) generate resized page thumbnail images in parallel",
            result_path="$.OCRPreproc",
        ).branch(*ocr_preproc_states)

        self.enrichment_step = SageMakerEnrichmentStep(
            self,
            "EnrichmentStep",
            lambda_role=self.shared_lambda_role,
            output_bucket=self.enriched_results_bucket,
            ssm_param_prefix=ssm_param_prefix,
            shared_sagemaker_caller_lambda=self.shared_sagemaker_lambda,
            textracted_input_jsonpath={
                "Bucket": sfn.JsonPath.string_at("$.OCRPreproc[0].Bucket"),
                "Key": sfn.JsonPath.string_at("$.OCRPreproc[0].Key"),
            },
            thumbnail_input_jsonpath={
                "Bucket": sfn.JsonPath.string_at("$.OCRPreproc[1].Bucket"),
                "Key": sfn.JsonPath.string_at("$.OCRPreproc[1].Key"),
            }
            if use_thumbnails
            else None,
        )
        self.postprocessing_step = LambdaPostprocStep(
            self,
            "PostProcessingStep",
            lambda_role=self.shared_lambda_role,
            ssm_param_prefix=ssm_param_prefix,
        )
        self.review_step = A2IReviewStep(
            self,
            "ReviewStep",
            lambda_role=self.shared_lambda_role,
            input_bucket=self.input_bucket,
            reviews_bucket=self.human_reviews_bucket,
            ssm_param_prefix=ssm_param_prefix,
        )

        success = sfn.Succeed(
            self,
            "Success",
            comment="All done! Well maybe in the real world you'd want to actually do something else...",
        )

        definition = (
            ocr_preproc.next(self.enrichment_step.sfn_task)
            .next(self.postprocessing_step.sfn_task)
            .next(
                sfn.Choice(self, "CheckConfidence")
                .when(
                    sfn.Condition.number_greater_than_equals("$.ModelResult.Confidence", 0.5),
                    success,
                )
                .otherwise(
                    self.review_step.sfn_task.next(success),
                )
            )
        )

        self.state_machine = sfn.StateMachine(
            self,
            "PipelineStateMachine",
            definition=definition,
            state_machine_type=sfn.StateMachineType.STANDARD,
            timeout=Duration.minutes(20),
        )
        self.shared_lambda_role.add_to_policy(
            PolicyStatement(
                sid="StateMachineNotify",
                actions=["states:SendTask*"],
                effect=Effect.ALLOW,
                resources=["*"],  # No resource types currently supported per the doc
            ),
        )

        self.trigger_lambda_role = Role(
            self,
            "S3PipelineTriggerLambdaRole",
            assumed_by=ServicePrincipal("lambda.amazonaws.com"),
            description="Execution role for S3 notification function triggering OCR pipeline",
            managed_policies=[
                ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                ),
                ManagedPolicy.from_aws_managed_policy_name("AWSXRayDaemonWriteAccess"),
            ],
        )
        self.input_bucket.grant_read(self.trigger_lambda_role)
        self.state_machine.grant_start_execution(self.trigger_lambda_role)
        self.trigger_lambda = PythonFunction(
            self,
            "S3PipelineTrigger",
            description="Trigger Step Functions OCR pipeline from S3 notification",
            entry=S3_TRIGGER_LAMBDA_PATH,
            environment={
                "STATE_MACHINE_ARN": self.state_machine.state_machine_arn,
            },
            index="main.py",
            handler="handler",
            memory_size=128,
            role=self.trigger_lambda_role,
            runtime=LambdaRuntime.PYTHON_3_9,
            timeout=Duration.seconds(15),
        )
        self.trigger_lambda.add_permission(
            "S3BucketNotification",
            action="lambda:InvokeFunction",
            source_account=Fn.sub("${AWS::AccountId}"),
            principal=ServicePrincipal("s3.amazonaws.com"),
        )
        self.input_bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.LambdaDestination(self.trigger_lambda),
        )

    @property
    def plain_textract_state_machine(self) -> sfn.StateMachine:
        """State maachine for running Textract only (rather than the full pipeline .state_machine)"""
        return self.ocr_step.textract_state_machine

    @property
    def sagemaker_endpoint_param(self) -> ssm.StringParameter:
        """SSM parameter linking the pipeline to a SageMaker enrichment model (endpoint name)"""
        return self.enrichment_step.endpoint_param

    @property
    def thumbnail_endpoint_param(self) -> Optional[ssm.StringParameter]:
        """SSM parameter linking the pipeline to a thumbnail generator endpoint (on SageMaker)

        This will be `None` if the construct was created with `use_thumbnails=False`
        """
        return None if self.thumbnails_step is None else self.thumbnails_step.endpoint_param

    @property
    def sagemaker_sns_topic(self) -> Optional[sns.Topic]:
        """SNS topic that async SageMaker endpoints should use for callback, or None if not enabled"""
        return self.enrichment_step.sfn_task.async_notify_topic

    @property
    def thumbnail_sns_topic(self) -> Optional[sns.Topic]:
        """SNS topic that async SageMaker endpoints should use for callback, or None if not enabled"""
        return (
            None
            if self.thumbnails_step is None
            else self.thumbnails_step.sfn_task.async_notify_topic
        )

    @property
    def entity_config_param(self) -> ssm.StringParameter:
        """SSM parameter defining the entity types configuration for rule-based post-processing"""
        return self.postprocessing_step.entity_config_param

    @property
    def review_a2i_role(self) -> Role:
        """IAM role to use for Amazon A2I human review flows in the pipeline"""
        return self.review_step.a2i_role

    @property
    def review_workflow_param(self) -> ssm.StringParameter:
        """SSM parameter defining the Amazon A2I workflow ARN for human reviews in the pipeline"""
        return self.review_step.workflow_param

    def config_read_statements(self, sid_prefix: Union[str, None] = "") -> List[PolicyStatement]:
        """Create PolicyStatements to grant read perms on pipeline configuration SSM parameters

        Arguments
        ---------
        sid_prefix : str | None
            Prefix to add to generated statement IDs for uniqueness, or "", or None to suppress
            SIDs.
        """
        return [
            SsmParameterReadStatement(
                resources=[
                    param
                    for param in (
                        self.sagemaker_endpoint_param,
                        self.thumbnail_endpoint_param,
                        self.entity_config_param,
                        self.review_workflow_param,
                    )
                    if param is not None
                ],
                sid=None if sid_prefix is None else (sid_prefix + "ReadPipelineConfigParams"),
            )
        ]

    def config_write_statements(self, sid_prefix: Union[str, None] = "") -> List[PolicyStatement]:
        """Create PolicyStatements to grant write perms on pipeline configuration SSM parameters

        Arguments
        ---------
        sid_prefix : str | None
            Prefix to add to generated statement IDs for uniqueness, or "", or None to suppress
            SIDs.
        """
        return [
            SsmParameterWriteStatement(
                resources=[
                    param
                    for param in (
                        self.sagemaker_endpoint_param,
                        self.thumbnail_endpoint_param,
                        self.entity_config_param,
                        self.review_workflow_param,
                    )
                    if param is not None
                ],
                sid=None if sid_prefix is None else (sid_prefix + "WritePipelineConfigParams"),
            ),
        ]

    def config_read_write_statements(
        self,
        sid_prefix: Union[str, None] = "",
    ) -> List[PolicyStatement]:
        """Create PolicyStatements to grant read & write perms on pipeline configuration SSM params

        Arguments
        ---------
        sid_prefix : str | None
            Prefix to add to generated statement IDs for uniqueness, or "", or None to suppress
            SIDs.
        """
        return self.config_read_statements(sid_prefix) + self.config_write_statements(sid_prefix)

    def sagemaker_model_statements(
        self,
        sid_prefix: Union[str, None] = "",
    ) -> List[PolicyStatement]:
        """Create PolicyStatements to grant required bucket permissions to SageMaker models

        Your SageMaker model/endpoint's execution role will need these permissions to be able to
        access the pipeline's intermediate buckets for Textract results and enriched results.

        Arguments
        ---------
        sid_prefix : str | None
            Prefix to add to generated statement IDs for uniqueness, or "", or None to suppress
            SIDs.
        """
        statements = [
            S3Statement(
                grant_write=False,
                resources=[self.textract_results_bucket],
                sid=None if sid_prefix is None else (sid_prefix + "ReadTextractBucket"),
            ),
            S3Statement(
                grant_write=True,
                resources=[self.enriched_results_bucket],
                sid=None if sid_prefix is None else (sid_prefix + "ReadWriteEnrichedBucket"),
            ),
            *self.enrichment_step.sagemaker_sns_statements(
                sid_prefix=None if sid_prefix is None else (sid_prefix + "Enrich"),
            ),
        ]
        if self.thumbnails_step is not None:
            statements += self.thumbnails_step.sagemaker_sns_statements(
                sid_prefix=None if sid_prefix is None else (sid_prefix + "Thumbs"),
            )
        return statements
