# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CDK for NLP/ML model enrichment stage of the OCR pipeline
"""
# Python Built-Ins:
from typing import Dict, List, Optional, Union

# External Dependencies:
from aws_cdk import Duration, Token
from aws_cdk.aws_iam import Effect, PolicyStatement, Role
from aws_cdk.aws_s3 import Bucket
import aws_cdk.aws_ssm as ssm
import aws_cdk.aws_stepfunctions as sfn
from constructs import Construct

# Local Dependencies:
from ..shared.sagemaker import SageMakerCallerFunction, SageMakerSSMStep


class SageMakerEnrichmentStep(Construct):
    """CDK construct for an OCR pipeline step to enrich Textract JSON on S3 using SageMaker

    This construct's `.sfn_task` takes input from JSONPath locations as specified by init params
    `textracted_input_jsonpath` (mandatory) and `thumbnail_input_jsonpath` (optional). The first
    links to a consolidated Textract JSON result in S3 as {Bucket, Key}. The second (if present),
    links to a consolidated page thumbnails file for the document: Again as S3 {Bucket, Key}. The
    task will set $.Textract on the output, with a similar { Bucket, Key } structure pointing to
    the enriched output JSON file.

    This step is implemented via AWS Lambda (rather than direct Step Function service call) to
    support looking up the configured SageMaker endpoint name from SSM within the same SFn step.

    When `support_async_endpoints` is enabled, the construct uses an asynchronous/TaskToken Lambda
    integration and checks at run-time whether the configured endpoint is sync or async. For async
    invocations, the same Lambda processes SageMaker callback events via SNS to notify SFn.
    """

    def __init__(
        self,
        scope: Construct,
        id: str,
        lambda_role: Role,
        output_bucket: Bucket,
        ssm_param_prefix: Union[Token, str],
        textracted_input_jsonpath: Dict[str, str],
        thumbnail_input_jsonpath: Optional[Dict[str, str]] = None,
        support_async_endpoints: bool = True,
        shared_sagemaker_caller_lambda: Optional[SageMakerCallerFunction] = None,
        **kwargs,
    ):
        """Create a SageMakerEnrichmentStep

        Parameters
        ----------
        lambda_role :
            IAM Execution Role for AWS Lambda, which will be used for the function to invoke the
            SageMaker endpoint.
        output_bucket :
            S3 Bucket where inference results should be stored.
        ssm_param_prefix :
            Name prefix under which the SSM SageMakerEndpointName parameter will be generated.
        textracted_input_jsonpath :
            Dict of `{ Bucket, Key }` locating the input document Textract result (should typically
            each be an `aws_stepfunctions.JsonPath` pointing to strings in the SFn state).
        thumbnail_input_jsonpath :
            Optional Dict of `{ Bucket, Key }` locating the thumbnail images archive for the input
            document (if thumbnailing is enabled). Structure as `textracted_input_jsonpath`.
        support_async_endpoints :
            As per `..shared.sagemaker.SageMakerSSMStep`
        shared_sagemaker_caller_lambda :
            Optional SageMakerCallerFunction Lambda for calling the SageMaker endpoint, if an
            already-created one is to be used (for sharing with other constructs).
        **kwargs :
            As per Construct parent
        """
        super().__init__(scope, id, **kwargs)

        self.endpoint_param = ssm.StringParameter(
            self,
            "EnrichmentSageMakerEndpointParam",
            description="Name of the SageMaker Endpoint to call for OCR result enrichment",
            parameter_name=f"{ssm_param_prefix}SageMakerEndpointName",
            simple_name=False,
            string_value="undefined",
        )
        lambda_role.add_to_policy(
            PolicyStatement(
                sid="ReadSageMakerEndpointParam",
                actions=["ssm:GetParameter"],
                effect=Effect.ALLOW,
                resources=[self.endpoint_param.parameter_arn],
            )
        )

        output_bucket.grant_read_write(lambda_role, "enriched/*")

        # Prepare the "Body" param for the Lambda function:
        inf_req_body = {
            "S3Input": textracted_input_jsonpath,
            "S3Output": {
                "Bucket": output_bucket.bucket_name,
                "Key": sfn.JsonPath.format(
                    "enriched/{}",
                    textracted_input_jsonpath["Key"],
                ),
            },
        }
        if thumbnail_input_jsonpath is None:
            # No need to upload this payload to S3: Lambda can directly invoke SageMaker on S3Input
            # if async, or pass the object if sync.
            body_upload = None
        else:
            inf_req_body["S3Thumbnails"] = thumbnail_input_jsonpath
            # Since our Body.S3Input doesn't contain the *entire* endpoint input (as the raw
            # Textract JSON is missing the S3Thumbnails link), to call an async SM endpoint Lambda
            # will need to upload the above `inf_req_body` JSON to S3 first:
            body_upload = {
                "Bucket": output_bucket.bucket_name,
                "Key": sfn.JsonPath.format(
                    "requests/{}",
                    textracted_input_jsonpath["Key"],
                ),
            }

        self.sfn_task = SageMakerSSMStep(
            self,
            "NLPEnrichmentModel",
            comment="Post-Process the Textract result with Amazon SageMaker",
            lambda_function=shared_sagemaker_caller_lambda,
            lambda_role=lambda_role,
            support_async_endpoints=support_async_endpoints,
            payload=sfn.TaskInput.from_object(
                {
                    # Because the caller lambda can be shared, we need to specify the param on req:
                    "EndpointNameParam": self.endpoint_param.parameter_name,
                    "Body": inf_req_body,
                    **({"BodyUpload": body_upload} if body_upload else {}),
                    "ContentType": "application/json",
                    **({"TaskToken": sfn.JsonPath.task_token} if support_async_endpoints else {}),
                }
            ),
            # We call the output variable 'Textract' here because it's an augmented Textract JSON -
            # so downstream components can treat it broadly as a Textract result:
            result_path="$.Textract",
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
        return self.sfn_task.sagemaker_sns_statements(sid_prefix=sid_prefix)
