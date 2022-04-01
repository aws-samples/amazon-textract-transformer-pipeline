# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CDK Construct for an OCR pipeline with field post-processing and human review
"""
# Python Built-Ins:
import os
from typing import List, Union

# External Dependencies:
from aws_cdk import Duration, Fn, RemovalPolicy, Stack, Token
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
import aws_cdk.aws_ssm as ssm
from constructs import Construct

# Local Dependencies:
from .enrichment import SageMakerEnrichmentStep
from .iam_utils import (
    S3Statement,
    SsmParameterReadStatement,
    SsmParameterWriteStatement,
)
from .ocr import TextractOcrStep
from .postprocessing import LambdaPostprocStep
from .review import A2IReviewStep

S3_TRIGGER_LAMBDA_PATH = os.path.join(os.path.dirname(__file__), "fn-trigger")


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
        **kwargs,
    ):
        """Create a ProcessingPipeline

        Arguments
        ---------
        scope : Construct
            CDK construct scope
        id : str
            CDK construct ID
        input_bucket : aws_cdk.aws_s3.Bucket
            The raw input bucket, to which this pipeline will attach and listen for documents being
            uploaded.
        ssm_param_prefix : Union[aws_cdk.Token, str]
            A prefix to apply to generated AWS SSM Parameter Store configuration parameter names,
            to help keep the account tidy. Should begin and end with a forward slash.
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
            "ProcessingPipelineLambdaRole",
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
        s3.Bucket.from_bucket_arn(
            self,
            "SageMakerDefaultBucket",
            f"arn:aws:s3:::sagemaker-{Stack.of(self).region}-{Stack.of(self).account}",
        ).grant_read_write(self.shared_lambda_role)

        self.ocr_step = TextractOcrStep(
            self,
            "OCRStep",
            lambda_role=self.shared_lambda_role,
        )
        self.enrichment_step = SageMakerEnrichmentStep(
            self,
            "EnrichmentStep",
            lambda_role=self.shared_lambda_role,
            output_bucket=self.enriched_results_bucket,
            ssm_param_prefix=ssm_param_prefix,
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
            self.ocr_step.sfn_task.next(self.enrichment_step.sfn_task)
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
            "ProcessingPipelineStateMachine",
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
            "ProcessingPipelineTriggerLambdaRole",
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
                "TEXTRACT_S3_BUCKET_NAME": self.textract_results_bucket.bucket_name,
                "TEXTRACT_S3_PREFIX": "textract",
            },
            index="main.py",
            handler="handler",
            memory_size=128,
            role=self.trigger_lambda_role,
            runtime=LambdaRuntime.PYTHON_3_8,
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
    def sagemaker_model_param(self) -> ssm.StringParameter:
        """SSM parameter linking the pipeline to a SageMaker enrichment model (endpoint name)"""
        return self.enrichment_step.model_param

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
                    self.sagemaker_model_param,
                    self.entity_config_param,
                    self.review_workflow_param,
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
                    self.sagemaker_model_param,
                    self.entity_config_param,
                    self.review_workflow_param,
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

    def enrichment_model_statements(
        self,
        sid_prefix: Union[str, None] = "",
    ) -> List[PolicyStatement]:
        """Create PolicyStatements to grant required bucket permissions to NLP enrichment model

        Your SageMaker model/endpoint's execution role will need these permissions to be able to
        access the pipeline's intermediate buckets for Textract results and enriched results.

        Arguments
        ---------
        sid_prefix : str | None
            Prefix to add to generated statement IDs for uniqueness, or "", or None to suppress
            SIDs.
        """
        return [
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
        ]
