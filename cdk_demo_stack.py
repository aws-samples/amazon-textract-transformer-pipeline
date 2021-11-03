# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""AWS CDK deployable stack for OCR pipeline sample
"""

# External Dependencies:
from aws_cdk import core as cdk
from aws_cdk.aws_iam import (
    ManagedPolicy,
    PolicyDocument,
)
import aws_cdk.aws_s3 as s3
import aws_cdk.aws_ssm as ssm

# Local Dependencies:
from annotation import AnnotationInfra
from pipeline import ProcessingPipeline
from pipeline.iam_utils import (
    S3Statement,
    SsmParameterReadStatement,
    StateMachineExecuteStatement,
)


class PipelineDemoStack(cdk.Stack):
    """Deployable CDK stack bundling the core OCR pipeline construct with supporting demo resources

    This stack bundles the core ProcessingPipeline construct with the additional resources required
    to deploy and use it for the demo: Such as the project ID SSM parameter, input data bucket,
    SageMaker permissions policy, and the infrastructure for custom annotation UIs in SageMaker
    Ground Truth.

    It also creates several CloudFormation stack outputs to help users find important created
    resources.
    """

    def __init__(
        self, scope: cdk.Construct, construct_id: str, default_project_id: str, **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Could consider just directly using the stack ID for this, but then if you were to vend
        # the stack through e.g. AWS Service Catalog you may not have control over setting a nice
        # readable stack ID:
        self.project_id_param = cdk.CfnParameter(
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
                + self.annotation_infra.get_data_science_policy_statements()
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
        self.sm_image_build_role_ssm_param = ssm.StringParameter(
            self,
            "SMImageBuildRoleSSMParam",
            string_value=self.annotation_infra.sm_image_build_role.role_name,
            description=(
                "Name of the CodeBuild execution role to use in SMStudio Image Build commands"
            ),
            parameter_name=f"/{self.project_id_param.value_as_string}/static/SMDockerBuildRole",
            simple_name=False,
        )
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
                    self.sm_image_build_role_ssm_param,
                    self.input_bucket_ssm_param,
                    self.reviews_bucket_ssm_param,
                    self.pipeline_statemachine_ssm_param,
                    self.textract_statemachine_ssm_param,
                    self.a2i_role_arn_param,
                ],
                sid="ReadStaticPipelineParams",
            ),
        )
