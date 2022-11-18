# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CDK for OCR stage of the document processing pipeline
"""
# Python Built-Ins:
from typing import List, Optional, Union

# External Dependencies:
from aws_cdk import Token
import aws_cdk.aws_iam as iam
from aws_cdk.aws_s3 import Bucket
from constructs import Construct

# Local Dependencies:
from .sagemaker_ocr import SageMakerOCRStep
from .textract_ocr import TextractOCRStep
from ..shared.sagemaker import SageMakerCallerFunction


class OCRStep(Construct):
    """CDK construct for a document pipeline step to OCR incoming documents/images

    This construct's `.sfn_task` expects inputs with $.Input.Bucket and $.Input.Key properties
    specifying the location of the raw input document, and will return an object with Bucket and
    Key pointing to a consolidated JSON OCR output in Amazon Textract-compatible format.

    In addition to the standard (Amazon Textract-based) option, this construct supports building
    and deploying alternative, custom OCR options. Multiple engines may be built and/or deployed (to
    support experimentation), but the pipeline must be pointed to exactly one custom SageMaker or
    Amazon Textract OCR provider.
    """

    def __init__(
        self,
        scope: Construct,
        id: str,
        lambda_role: iam.Role,
        ssm_param_prefix: Union[Token, str],
        input_bucket: Bucket,
        output_bucket: Bucket,
        output_prefix: str,
        input_prefix: Optional[str] = None,
        build_sagemaker_ocrs: List[str] = [],
        deploy_sagemaker_ocrs: List[str] = [],
        use_sagemaker_ocr: Optional[str] = None,
        enable_sagemaker_autoscaling: bool = False,
        shared_sagemaker_caller_lambda: Optional[SageMakerCallerFunction] = None,
    ):
        """Create an OCRStep

        Parameters
        ----------
        scope :
            CDK construct scope
        id :
            CDK construct ID
        lambda_role :
            IAM Role that the Amazon Textract-invoking Lambda function will run with
        ssm_param_prefix :
            Prefix to be applied to generated SSM pipeline configuration parameter names (including
            the parameter to configure SageMaker endpoint name for thumbnail generation).
        input_bucket :
            Bucket from which input documents will be fetched. If auto-deployment of a thumbnailer
            endpoint is enabled, the model execution role will be granted access to this bucket
            (limited to `input_prefix`).
        output_bucket :
            (Pre-existing) S3 bucket where Textract result files should be stored
        output_prefix :
            Prefix under which Textract result files should be stored in S3 (under this prefix,
            the original input document keys will be mapped).
        input_prefix :
            Prefix under `input_bucket` from which input documents will be fetched. Used to
            configure SageMaker model execution role permissions when auto-deployment of thumbnailer
            endpoint is enabled.
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
        enable_sagemaker_autoscaling :
            Set True to enable auto-scaling on SageMaker OCR endpoints (if any are deployed), to
            optimize resource usage (recommended for production use). Set False to disable it and
            avoid cold-starts (good for development).
        shared_sagemaker_caller_lambda :
            Optional pre-existing SageMaker caller Lambda function, to share this between multiple
            SageMakerSSMSteps in the app if required.
        """
        super().__init__(scope, id)

        if len(build_sagemaker_ocrs) > 0:
            self.sagemaker_step = SageMakerOCRStep(
                self,
                "SageMakerStep",
                lambda_role=lambda_role,
                ssm_param_prefix=ssm_param_prefix,
                input_bucket=input_bucket,
                ocr_results_bucket=output_bucket,
                input_prefix=input_prefix,
                ocr_results_prefix=output_prefix,
                build_engine_names=build_sagemaker_ocrs,
                deploy_engine_names=deploy_sagemaker_ocrs,
                use_engine_name=use_sagemaker_ocr,
                enable_autoscaling=enable_sagemaker_autoscaling,
                shared_sagemaker_caller_lambda=shared_sagemaker_caller_lambda,
            )
        else:
            self.sagemaker_step = None

        self.textract_step = TextractOCRStep(
            self,
            "TextractStep",
            lambda_role=lambda_role,
            output_bucket=output_bucket,
            output_prefix=output_prefix,
        )

        if use_sagemaker_ocr and self.sagemaker_step:
            self.sfn_task = self.sagemaker_step.sfn_task
        else:
            self.sfn_task = self.textract_step.sfn_task

    @property
    def textract_state_machine(self):
        return self.textract_step.textract_state_machine
