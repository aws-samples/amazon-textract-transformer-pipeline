# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CDK for NLP/ML model enrichment stage of the OCR pipeline
"""
# Python Built-Ins:
import os
from typing import Union

# External Dependencies:
from aws_cdk import core as cdk
from aws_cdk.aws_iam import Effect, PolicyStatement, Role
from aws_cdk.aws_lambda import Runtime as LambdaRuntime
from aws_cdk.aws_lambda_python import PythonFunction
from aws_cdk.aws_s3 import Bucket
import aws_cdk.aws_ssm as ssm
import aws_cdk.aws_stepfunctions as sfn
import aws_cdk.aws_stepfunctions_tasks as sfn_tasks


SM_LAMBDA_PATH = os.path.join(os.path.dirname(__file__), "fn-call-sagemaker")


class SageMakerEnrichmentStep(cdk.Construct):
    """CDK construct for an OCR pipeline step to enrich Textract JSON on S3 using SageMaker

    This construct's `.sfn_task` expects inputs with $.Textract.Bucket and $.Textract.Key
    properties, and will override that $.Textract field on the output with the bucket and key for
    the enriched output JSON file.
    """
    def __init__(
        self,
        scope: cdk.Construct,
        id: str,
        lambda_role: Role,
        output_bucket: Bucket,
        ssm_param_prefix: Union[cdk.Token, str],
        **kwargs,
    ):
        super().__init__(scope, id, **kwargs)
        lambda_role.add_to_policy(
            PolicyStatement(
                sid="InvokeSageMakerEndpoints",
                actions=["sagemaker:InvokeEndpoint"],
                effect=Effect.ALLOW,
                resources=["*"],
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
            },
            role=lambda_role,
            runtime=LambdaRuntime.PYTHON_3_8,
            timeout=cdk.Duration.seconds(60),
        )

        self.sfn_task = sfn_tasks.LambdaInvoke(
            self,
            "NLPEnrichmentModel",
            comment="Post-Process the Textract result with Amazon SageMaker",
            lambda_function=self.caller_lambda,
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
                }
            ),
            payload_response_only=True,
            result_path="$.Textract",  # Overwrite the Textract output state
        )
