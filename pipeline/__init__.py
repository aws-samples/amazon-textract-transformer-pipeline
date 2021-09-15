# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CDK Construct and Stack for an OCR pipeline with field post-processing and human review
"""
# Python Built-Ins:
import os
from typing import List, Union

# External Dependencies:
from aws_cdk import core as cdk
from aws_cdk.aws_iam import (
    Effect,
    ManagedPolicy,
    PolicyDocument,
    PolicyStatement,
    Role,
    ServicePrincipal,
)
from aws_cdk.aws_lambda import Runtime as LambdaRuntime
from aws_cdk.aws_lambda_python import PythonFunction
import aws_cdk.aws_s3 as s3
import aws_cdk.aws_s3_notifications as s3n
import aws_cdk.aws_stepfunctions as sfn
import aws_cdk.aws_ssm as ssm

# Local Dependencies:
from .enrichment import SageMakerEnrichmentStep
from .iam_utils import (
    S3Statement,
    SsmParameterReadStatement,
    SsmParameterWriteStatement,
    StateMachineExecuteStatement,
)
from .ocr import TextractOcrStep
from .postprocessing import LambdaPostprocStep
from .review import A2IReviewStep

S3_TRIGGER_LAMBDA_PATH = os.path.join(os.path.dirname(__file__), "fn-trigger")


class ProcessingPipeline(cdk.Construct):
    """CDK construct for an OCR pipeline with field post-processing and human review

    This is the main top-level construct for the sample solution, implementing an AWS Step
    Functions-based pipeline orchestrating the different stages of the pipeline.
    """
    def __init__(
        self,
        scope: cdk.Construct,
        id: str,
        input_bucket: s3.Bucket,
        ssm_param_prefix: Union[cdk.Token, str],
        **kwargs,
    ):
        """Create a ProcessingPipeline

        Arguments
        ---------
        scope : cdk.Construct
            CDK construct scope
        id : str
            CDK construct ID
        input_bucket : aws_cdk.aws_s3.Bucket
            The raw input bucket, to which this pipeline will attach and listen for documents being
            uploaded.
        ssm_param_prefix : Union[aws_cdk.core.Token, str]
            A prefix to apply to generated AWS SSM Parameter Store configuration parameter names,
            to help keep the account tidy. Should begin and end with a forward slash.
        **kwargs : Any
            Passed through to parent cdk.Construct
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
            lifecycle_rules=[s3.LifecycleRule(enabled=True, expiration=cdk.Duration.days(7))],
            removal_policy=cdk.RemovalPolicy.DESTROY,
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
            lifecycle_rules=[s3.LifecycleRule(enabled=True, expiration=cdk.Duration.days(7))],
            removal_policy=cdk.RemovalPolicy.DESTROY,
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
            lifecycle_rules=[s3.LifecycleRule(enabled=True, expiration=cdk.Duration.days(7))],
            removal_policy=cdk.RemovalPolicy.DESTROY,
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
            f"arn:aws:s3:::sagemaker-{cdk.Stack.of(self).region}-{cdk.Stack.of(self).account}",
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
            timeout=cdk.Duration.minutes(20),
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
            timeout=cdk.Duration.seconds(15),
        )
        self.trigger_lambda.add_permission(
            "S3BucketNotification",
            action="lambda:InvokeFunction",
            source_account=cdk.Fn.sub("${AWS::AccountId}"),
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


class ProcessingPipelineStack(cdk.Stack):
    """Deployable CDK stack for a demo OCR pipeline with field post-processing and human review

    This stack bundles the core Pipeline construct with the additional resources required to deploy
    and use it for the demo: Such as the input bucket, and SageMaker permissions policy.
    """
    def __init__(self, scope: cdk.Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.project_id_param = cdk.CfnParameter(
            self,
            "ProjectId",
            allowed_pattern=r"[a-zA-Z0-9]+(\-[a-zA-Z0-9]+)*",
            constraint_description="Alphanumeric with internal hyphens allowed",
            default="ocr-transformers-demo",
            description=(
                "ID to look up this stack's resources from SageMaker notebooks, used in folder "
                "prefixes for SSM parameters."
            ),
            max_length=25,
            min_length=3,
        )

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
                s3.LifecycleRule(enabled=True, expiration=cdk.Duration.days(7)),
            ],
            removal_policy=cdk.RemovalPolicy.DESTROY,
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
                + self.pipeline.config_read_write_statements()
                # In the notebooks we'll use the same execution role for the trained model/endpoint
                # as the notebook itself runs with - so need to grant the role the required perms
                # for reading/writing relevant S3 buckets in the pipeline:
                + self.pipeline.enrichment_model_statements(),
            ),
        )

        # We'd like the stack to push some useful outputs to CloudFormation, and defining them here
        # rather than in the lower-level constructs will keep the constructs flexible/reusable.
        #
        # We override the auto-generated logical IDs to make the names simple to find in console.
        self.data_science_policy_output = cdk.CfnOutput(
            self,
            "DataSciencePolicyName",
            description=(
                "Name of the IAM policy with permissions needed for the SageMaker notebooks to "
                "access this stack's resources. Add this policy to your SageMaker execution role."
            ),
            value=self.data_science_policy.managed_policy_name,
        )
        self.data_science_policy_output.override_logical_id("DataSciencePolicyName")
        self.input_bucket_name_output = cdk.CfnOutput(
            self,
            "InputBucketName",
            description="Name of the S3 bucket to which input documents should be uploaded",
            value=self.pipeline.input_bucket.bucket_name,
        )
        self.input_bucket_name_output.override_logical_id("InputBucketName")
        self.pipeline_statemachine_output = cdk.CfnOutput(
            self,
            "PipelineStateMachine",
            description="ARN of the State Machine for the end-to-end OCR pipeline",
            value=self.pipeline.state_machine.state_machine_arn,
        )
        self.pipeline_statemachine_output.override_logical_id("PipelineStateMachine")
        self.textract_statemachine_output = cdk.CfnOutput(
            self,
            "PlainTextractStateMachine",
            description="ARN of the State Machine for *only* running Textract (no enrichments)",
            value=self.pipeline.plain_textract_state_machine.state_machine_arn,
        )
        self.textract_statemachine_output.override_logical_id("PlainTextractStateMachine")
        self.model_param_output = cdk.CfnOutput(
            self,
            "SageMakerEndpointParamName",
            description="SSM parameter to configure the pipeline's SageMaker endpoint name",
            value=self.pipeline.sagemaker_model_param.parameter_name,
        )
        self.model_param_output.override_logical_id("SageMakerEndpointParamName")
        self.entity_config_param_output = cdk.CfnOutput(
            self,
            "EntityConfigParamName",
            description=(
                "JSON configuration describing the field types to be extracted by the pipeline"
            ),
            value=self.pipeline.entity_config_param.parameter_name,
        )
        self.entity_config_param_output.override_logical_id("EntityConfigParamName")
        self.a2i_role_arn_output = cdk.CfnOutput(
            self,
            "A2IHumanReviewExecutionRoleArn",
            description="ARN of the execution Role to use for Amazon A2I human review workflows",
            value=self.pipeline.review_a2i_role.role_arn,
        )
        self.a2i_role_arn_output.override_logical_id("A2IHumanReviewExecutionRoleArn")
        self.workflow_param_output = cdk.CfnOutput(
            self,
            "A2IHumanReviewFlowParamName",
            description="SSM parameter to configure the pipeline's A2I review workflow ARN",
            value=self.pipeline.review_workflow_param.parameter_name,
        )
        self.workflow_param_output.override_logical_id("A2IHumanReviewFlowParamName")
        self.reviews_bucket_name_output = cdk.CfnOutput(
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
        self.input_bucket_ssm_param = ssm.StringParameter(
            self,
            "InputBucketNameSSMParam",
            string_value=self.input_bucket.bucket_name,
            description="Name of the S3 bucket to which input documents should be uploaded",
            parameter_name=f"/{self.project_id_param.value_as_string}/static/InputBucket",
            simple_name=False,
        )
        self.reviews_bucket_ssm_param = ssm.StringParameter(
            self,
            "ReviewsBucketNameSSMParam",
            string_value=self.pipeline.human_reviews_bucket.bucket_name,
            description="Name of the S3 bucket to which human reviews should be stored",
            parameter_name=f"/{self.project_id_param.value_as_string}/static/ReviewsBucket",
            simple_name=False,
        )
        self.pipeline_statemachine_ssm_param = ssm.StringParameter(
            self,
            "PipelineStateMachineSSMParam",
            string_value=self.pipeline.state_machine.state_machine_arn,
            description="ARN of the State Machine for the end-to-end OCR pipeline",
            parameter_name=(
                f"/{self.project_id_param.value_as_string}/static/PipelineStateMachine"
            ),
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
        self.a2i_role_arn_param = ssm.StringParameter(
            self,
            "A2IExecutionRoleArnParam",
            string_value=self.pipeline.review_a2i_role.role_arn,
            description="ARN of the execution role which A2I human review workflows should use",
            parameter_name=f"/{self.project_id_param.value_as_string}/static/A2IExecutionRoleArn",
            simple_name=False,
        )

        self.data_science_policy.add_statements(
            SsmParameterReadStatement(
                resources=[
                    self.input_bucket_ssm_param,
                    self.reviews_bucket_ssm_param,
                    self.pipeline_statemachine_ssm_param,
                    self.textract_statemachine_ssm_param,
                    self.a2i_role_arn_param,
                ],
                sid="ReadStaticPipelineParams",
            ),
        )
