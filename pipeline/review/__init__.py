# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CDK for human review stage of the OCR pipeline
"""
# Python Built-Ins:
import os
from typing import Union

# External Dependencies:
from aws_cdk import core as cdk
from aws_cdk.aws_iam import Effect, PolicyStatement, Role, ServicePrincipal
from aws_cdk.aws_lambda import Runtime as LambdaRuntime
from aws_cdk.aws_lambda_python import PythonFunction
from aws_cdk.aws_s3 import Bucket, EventType
import aws_cdk.aws_s3_notifications as s3n
import aws_cdk.aws_ssm as ssm
import aws_cdk.aws_stepfunctions as sfn
import aws_cdk.aws_stepfunctions_tasks as sfn_tasks


START_REVIEW_LAMBDA_PATH = os.path.join(os.path.dirname(__file__), "fn-start-review")
REVIEW_CALLBACK_LAMBDA_PATH = os.path.join(os.path.dirname(__file__), "fn-review-callback")


class A2IReviewStep(cdk.Construct):
    """CDK construct for an OCR pipeline step to have humans review extracted document fields

    This construct's `.sfn_task` expects inputs with a $.ModelResult object and an input document
    specified by $.Input.Bucket and $.Input.Key. An Amazon A2I Human Review Loop will be triggered
    to manually review the model's prediction, and the Step Function execution will be resumed when
    the review is complete, with an updated $.ModelResult in the output state.
    """
    def __init__(
        self,
        scope: cdk.Construct,
        id: str,
        lambda_role: Role,
        input_bucket: Bucket,
        reviews_bucket: Bucket,
        ssm_param_prefix: Union[cdk.Token, str],
        **kwargs,
    ):
        super().__init__(scope, id, **kwargs)

        self.workflow_param = ssm.StringParameter(
            self,
            "A2IHumanReviewFlowParam",
            description="ARN of the Amazon A2I workflow definition to call for human reviews",
            parameter_name=f"{ssm_param_prefix}HumanReviewFlowArn",
            simple_name=False,
            string_value="undefined",
            type=ssm.ParameterType.STRING,
        )
        lambda_role.add_to_policy(
            PolicyStatement(
                sid="ReadSSMWorkflowParam",
                actions=["ssm:GetParameter"],
                effect=Effect.ALLOW,
                resources=[self.workflow_param.parameter_arn],
            )
        )
        lambda_role.add_to_policy(
            PolicyStatement(
                sid="StartAnyA2IHumanLoop",
                actions=["sagemaker:StartHumanLoop"],
                effect=Effect.ALLOW,
                resources=["*"],
            )
        )

        self.start_lambda = PythonFunction(
            self,
            "StartHumanReview",
            description="Kick off A2I human review for OCR processing pipeline",
            entry=START_REVIEW_LAMBDA_PATH,
            environment={
                "DEFAULT_FLOW_DEFINITION_ARN_PARAM": self.workflow_param.parameter_name,
            },
            index="main.py",
            handler="handler",
            memory_size=128,
            role=lambda_role,
            runtime=LambdaRuntime.PYTHON_3_8,
            timeout=cdk.Duration.seconds(10),
        )
        self.callback_lambda = PythonFunction(
            self,
            "HumanReviewCallback",
            description="Return A2I human review result to OCR processing pipeline Step Function",
            entry=REVIEW_CALLBACK_LAMBDA_PATH,
            index="main.py",
            handler="handler",
            memory_size=128,
            role=lambda_role,
            runtime=LambdaRuntime.PYTHON_3_8,
            timeout=cdk.Duration.seconds(60),
        )
        self.a2i_role = Role(
            self,
            "ProcessingPipelineA2IRole",
            assumed_by=ServicePrincipal("sagemaker.amazonaws.com"),
            description="Execution Role for Amazon A2I human review workflows",
        )
        input_bucket.grant_read(self.a2i_role)
        reviews_bucket.grant_read_write(self.a2i_role)
        Bucket.from_bucket_arn(
            self,
            "SageMakerDefaultBucket",
            f"arn:aws:s3:::sagemaker-{cdk.Stack.of(self).region}-{cdk.Stack.of(self).account}",
        ).grant_read_write(self.a2i_role)

        reviews_bucket.add_event_notification(
            dest=s3n.LambdaDestination(self.callback_lambda),
            event=EventType.OBJECT_CREATED,
        )

        self.sfn_task = sfn_tasks.LambdaInvoke(
            self,
            "HumanReview",
            comment="Run an Amazon A2I human loop to review the annotations manually",
            lambda_function=self.start_lambda,
            integration_pattern=sfn.IntegrationPattern.WAIT_FOR_TASK_TOKEN,
            payload=sfn.TaskInput.from_object(
                {
                    "ModelResult.$": "$.ModelResult",
                    # TODO: Can we add a pass state with Parameters to filter the inputs?
                    "TaskObject": {
                        "Bucket": sfn.JsonPath.string_at("$.Input.Bucket"),
                        "Key": sfn.JsonPath.string_at("$.Input.Key"),
                    },
                    "TaskToken": sfn.JsonPath.task_token,
                }
            ),
            result_path="$.ModelResult",
            timeout=cdk.Duration.minutes(20),
        )
