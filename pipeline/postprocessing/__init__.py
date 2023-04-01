# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CDK for rule-based post-processing stage of the OCR pipeline
"""
# Python Built-Ins:
import json
from typing import Union

# External Dependencies:
from aws_cdk import Duration, Token
from aws_cdk.aws_iam import Effect, PolicyStatement, Role
from aws_cdk.aws_lambda import Runtime as LambdaRuntime
from aws_cdk.aws_lambda_python_alpha import PythonFunction
import aws_cdk.aws_ssm as ssm
import aws_cdk.aws_stepfunctions as sfn
import aws_cdk.aws_stepfunctions_tasks as sfn_tasks
from constructs import Construct

# Local Dependencies:
from ..shared import abs_path


POSTPROC_LAMBDA_PATH = abs_path("fn-postprocess", __file__)

# Not technically necessary as the notebook guides users to configure this through AWS SSM, but
# useful to set the defaults per the notebook for speedy setup:
DEFAULT_ENTITY_CONFIG = [
    {
        "ClassId": 0,
        "Name": "Agreement Effective Date",
        "Optional": True,
        "Select": "first",
    },
    {
        "ClassId": 1,
        "Name": "APR - Introductory",
        "Optional": True,
        "Select": "confidence",
    },
    {
        "ClassId": 2,
        "Name": "APR - Balance Transfers",
        "Optional": True,
        "Select": "confidence",
    },
    {
        "ClassId": 3,
        "Name": "APR - Cash Advances",
        "Optional": True,
        "Select": "confidence",
    },
    {
        "ClassId": 4,
        "Name": "APR - Purchases",
        "Optional": True,
        "Select": "confidence",
    },
    {
        "ClassId": 5,
        "Name": "APR - Penalty",
        "Optional": True,
        "Select": "confidence",
    },
    {
        "ClassId": 6,
        "Name": "APR - General",
        "Optional": True,
        "Select": "confidence",
    },
    {
        "ClassId": 7,
        "Name": "APR - Other",
        "Optional": True,
        "Select": "confidence",
    },
    {
        "ClassId": 8,
        "Name": "Fee - Annual",
        "Optional": True,
        "Select": "confidence",
    },
    {
        "ClassId": 9,
        "Name": "Fee - Balance Transfer",
        "Optional": True,
        "Select": "confidence",
    },
    {
        "ClassId": 10,
        "Name": "Fee - Late Payment",
        "Optional": True,
        "Select": "confidence",
    },
    {
        "ClassId": 11,
        "Name": "Fee - Returned Payment",
        "Optional": True,
        "Select": "confidence",
    },
    {
        "ClassId": 12,
        "Name": "Fee - Foreign Transaction",
        "Optional": True,
        "Select": "shortest",
    },
    {
        "ClassId": 13,
        "Name": "Fee - Other",
        "Ignore": True,
    },
    {
        "ClassId": 14,
        "Name": "Card Name",
    },
    {
        "ClassId": 15,
        "Name": "Provider Address",
        "Optional": True,
        "Select": "confidence",
    },
    {
        "ClassId": 16,
        "Name": "Provider Name",
        "Select": "longest",
    },
    {
        "ClassId": 17,
        "Name": "Min Payment Calculation",
        "Ignore": True,
    },
    {
        "ClassId": 18,
        "Name": "Local Terms",
        "Ignore": True,
    },
]


class LambdaPostprocStep(Construct):
    """CDK construct for an OCR pipeline step consolidate document fields from enriched OCR JSON

    This construct's `.sfn_task` expects inputs with $.Textract.Bucket and $.Textract.Key
    properties, and will process this object with a Lambda function to add a $.ModelResult object
    to the output state: Consolidating detections of the different fields as defined by the
    field/entity configuration JSON in AWS SSM.
    """

    def __init__(
        self,
        scope: Construct,
        id: str,
        lambda_role: Role,
        ssm_param_prefix: Union[Token, str],
        **kwargs,
    ):
        super().__init__(scope, id, **kwargs)
        self.entity_config_param = ssm.StringParameter(
            self,
            "EntityConfigParam",
            description=(
                "JSON configuration describing the field types to be extracted by the pipeline"
            ),
            parameter_name=f"{ssm_param_prefix}EntityConfiguration",
            simple_name=False,
            string_value=json.dumps(DEFAULT_ENTITY_CONFIG, indent=2),
        )
        lambda_role.add_to_policy(
            PolicyStatement(
                sid="ReadSSMEntityConfigParam",
                actions=["ssm:GetParameter"],
                effect=Effect.ALLOW,
                resources=[self.entity_config_param.parameter_arn],
            )
        )
        self.caller_lambda = PythonFunction(
            self,
            "PostProcessFn",
            description="Post-process SageMaker-enriched Textract JSON to extract business fields",
            entry=POSTPROC_LAMBDA_PATH,
            environment={
                "DEFAULT_ENTITY_CONFIG_PARAM": self.entity_config_param.parameter_name,
            },
            index="main.py",
            handler="handler",
            memory_size=1024,
            role=lambda_role,
            runtime=LambdaRuntime.PYTHON_3_9,
            timeout=Duration.seconds(120),
        )

        self.sfn_task = sfn_tasks.LambdaInvoke(
            self,
            "PostProcess",
            comment="Post-Process the enriched Textract data to your business-level fields",
            lambda_function=self.caller_lambda,
            payload=sfn.TaskInput.from_object(
                {
                    "Input": {
                        "Bucket": sfn.JsonPath.string_at("$.Textract.Bucket"),
                        "Key": sfn.JsonPath.string_at("$.Textract.Key"),
                    },
                }
            ),
            payload_response_only=True,
            result_path="$.ModelResult",
        )
