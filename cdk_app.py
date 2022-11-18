#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""AWS CDK app entry point for OCR pipeline sample
"""
# Python Built-Ins:
import json
import os

# External Dependencies:
from aws_cdk import App

# Local Dependencies:
from cdk_demo_stack import PipelineDemoStack
from pipeline.config_utils import bool_env_var, list_env_var


# Top-level configurations are loaded from environment variables at the point `cdk synth` or
# `cdk deploy` is run (or you can override here):
config = {
    # Used as a prefix for some cloud resources e.g. SSM parameters:
    "default_project_id": os.environ.get("DEFAULT_PROJECT_ID", default="ocr-transformers-demo"),

    # Set False to skip deploying the page thumbnail image generator, if you're only using models
    # (like LayoutLMv1) that don't take page image as input features:
    "use_thumbnails": bool_env_var("USE_THUMBNAILS", default=True),

    # Set True to enable auto-scale-to-zero on auto-deployed SageMaker endpoints (including the
    # thumbnail generator and any custom OCR engines). This saves costs for low-volume workloads,
    # but introduces a few minutes' extra cold start for requests when all instances are released:
    "enable_sagemaker_autoscaling": bool_env_var("ENABLE_SM_AUTOSCALING", default=False),

    # To use alternative Tesseract OCR instead of Amazon Textract, before running `cdk deploy` run:
    #   export BUILD_SM_OCRS=tesseract
    #   export DEPLOY_SM_OCRS=tesseract
    #   export USE_SM_OCR=tesseract
    # ...Or edit the defaults below to `["tesseract"]` and `"tesseract"`
    "build_sagemaker_ocrs": list_env_var("BUILD_SM_OCRS", default=[]),
    "deploy_sagemaker_ocrs": list_env_var("DEPLOY_SM_OCRS", default=[]),
    "use_sagemaker_ocr": os.environ.get("USE_SM_OCR", default=None),
}

app = App()
print(f"Deploying stack with configuration:\n{json.dumps(config, indent=2)}")
demo_stack = PipelineDemoStack(
    app,
    "OCRPipelineDemo",
    **config,
)
app.synth()
