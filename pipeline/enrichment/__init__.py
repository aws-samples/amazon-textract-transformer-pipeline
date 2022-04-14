# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CDK for NLP/ML model enrichment stage of the OCR pipeline
"""
# Python Built-Ins:
import os
from typing import List, Optional, Union

# External Dependencies:
from aws_cdk import Duration, Token
from aws_cdk.aws_iam import Effect, PolicyStatement, Role
from aws_cdk.aws_lambda import Runtime as LambdaRuntime
from aws_cdk.aws_lambda_python_alpha import PythonFunction
from aws_cdk.aws_s3 import Bucket
import aws_cdk.aws_sns as sns
import aws_cdk.aws_sns_subscriptions as subs
import aws_cdk.aws_ssm as ssm
import aws_cdk.aws_stepfunctions as sfn
import aws_cdk.aws_stepfunctions_tasks as sfn_tasks
from constructs import Construct


SM_LAMBDA_PATH = os.path.join(os.path.dirname(__file__), "fn-call-sagemaker")


class SageMakerEnrichmentStep(Construct):
    """CDK construct for an OCR pipeline step to enrich Textract JSON on S3 using SageMaker

    This construct's `.sfn_task` expects inputs with $.Textract.Bucket and $.Textract.Key
    properties, and will override that $.Textract field on the output with the bucket and key for
    the enriched output JSON file.

    This step is implemented via AWS Lambda (rather than direct Step Function service call) to
    support looking up the configured SageMaker endpoint name from SSM within the same SFn step.

    When `support_async_endpoints` is enabled, the construct uses an asynchronous/TaskToken Lambda
    integration and checks at run-time whether the configured endpoint is sync or async. For async
    invocations, the same Lambda processes SageMaker callback events via SNS to notify SFn.
    """

    async_notify_topic: Optional[sns.Topic]

    def __init__(
        self,
        scope: Construct,
        id: str,
        lambda_role: Role,
        output_bucket: Bucket,
        ssm_param_prefix: Union[Token, str],
        support_async_endpoints: bool = True,
        **kwargs,
    ):
        super().__init__(scope, id, **kwargs)
        lambda_role.add_to_policy(
            PolicyStatement(
                sid="SageMakerEndpointsDescribeInvoke",
                actions=["sagemaker:InvokeEndpoint"]
                + (
                    ["sagemaker:DescribeEndpoint", "sagemaker:InvokeEndpointAsync"]
                    if support_async_endpoints
                    else []
                ),
                effect=Effect.ALLOW,
                resources=["*"],
            )
        )
        if support_async_endpoints:
            lambda_role.add_to_policy(
                PolicyStatement(
                    sid="StateMachineNotify",
                    actions=["states:SendTask*"],
                    effect=Effect.ALLOW,
                    resources=["*"],  # No resource types currently supported per the doc
                )
            )

        self.model_param = ssm.StringParameter(
            self,
            "EnrichmentSageMakerEndpointParam",
            description="Name of the SageMaker Endpoint to call for OCR result enrichment",
            parameter_name=f"{ssm_param_prefix}SageMakerEndpointName",
            simple_name=False,
            string_value="undefined",
            type=ssm.ParameterType.STRING,
        )
        lambda_role.add_to_policy(
            PolicyStatement(
                sid="ReadSSMModelParam",
                actions=["ssm:GetParameter"],
                effect=Effect.ALLOW,
                resources=[self.model_param.parameter_arn],
            )
        )

        self.caller_lambda = PythonFunction(
            self,
            "CallSageMakerModel",
            entry=SM_LAMBDA_PATH,
            index="main.py",
            handler="handler",
            memory_size=128,
            environment={
                "DEFAULT_ENDPOINT_NAME_PARAM": self.model_param.parameter_name,
                "SUPPORT_ASYNC_ENDPOINTS": "1" if support_async_endpoints else "0",
            },
            role=lambda_role,
            runtime=LambdaRuntime.PYTHON_3_8,
            timeout=Duration.seconds(60),
        )

        if support_async_endpoints:
            self.async_notify_topic = sns.Topic(self, "SageMakerAsyncTopic")
            self.async_notify_topic.add_subscription(subs.LambdaSubscription(self.caller_lambda))
        else:
            self.async_notify_topic = None

        self.sfn_task = sfn_tasks.LambdaInvoke(
            self,
            "NLPEnrichmentModel",
            comment="Post-Process the Textract result with Amazon SageMaker",
            lambda_function=self.caller_lambda,
            integration_pattern=(
                sfn.IntegrationPattern.WAIT_FOR_TASK_TOKEN
                if support_async_endpoints
                else sfn.IntegrationPattern.REQUEST_RESPONSE
            ),
            payload=sfn.TaskInput.from_object(
                {
                    "Body": {
                        "S3Input": {
                            "Bucket": sfn.JsonPath.string_at("$.Textract.Bucket"),
                            "Key": sfn.JsonPath.string_at("$.Textract.Key"),
                        },
                        "S3Output": {
                            "Bucket": output_bucket.bucket_name,
                            "Key": sfn.JsonPath.string_at("$.Textract.Key"),
                        },
                    },
                    # Explicit ContentType is necessary for async integration at time of writing,
                    # because otherwise the endpoint mis-recognises the request as octet-stream:
                    "ContentType": "application/json",
                    **({"TaskToken": sfn.JsonPath.task_token} if support_async_endpoints else {}),
                }
            ),
            payload_response_only=None if support_async_endpoints else True,
            result_path="$.Textract",  # Overwrite the Textract output state
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
        if not self.async_notify_topic:
            return []

        return [
            PolicyStatement(
                actions=["sns:Publish"],
                effect=Effect.ALLOW,
                resources=[self.async_notify_topic.topic_arn],
                sid=None if sid_prefix is None else (sid_prefix + "PublishSageMakerTopic"),
            )
        ]
