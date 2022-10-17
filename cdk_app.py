#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""AWS CDK app entry point for OCR pipeline sample
"""
# Python Built-Ins:
import json
import os
from typing import List, Optional

# External Dependencies:
from aws_cdk import App

# Local Dependencies:
from cdk_demo_stack import PipelineDemoStack


def bool_env_var(env_var_name: str, default: Optional[bool] = None) -> bool:
    """Parse a boolean environment variable"""
    raw = os.environ.get(env_var_name)
    if raw is None:
        if default is None:
            raise ValueError(f"Mandatory boolean env var '{env_var_name}' not found")
        return default
    raw = raw.lower()
    if raw in ("1", "true", "y", "yes"):
        return True
    elif raw in ("", "0", "false", "n", "no"):
        return False
    else:
        raise ValueError(
            "Couldn't interpret env var '%s' as boolean. Got: '%s'"
            % (env_var_name, raw)
        )


def list_env_var(env_var_name: str, default: Optional[List[str]] = None) -> List[str]:
    """Parse a comma-separated string list from an environment variable"""
    raw = os.environ.get(env_var_name)
    if raw is None:
        if default is None:
            raise ValueError(f"Mandatory string-list env var {env_var_name} not found")
        return default[:]
    return [s for s in raw.split(",") if s]  # (list comp as "" should yield [], not [""])


# Top-level configurations loaded from environment variables (or override here):
config = {
    "default_project_id": os.environ.get("DEFAULT_PROJECT_ID", default="ocr-transformers-demo"),
    "use_thumbnails": bool_env_var("USE_THUMBNAILS", default=True),
    "enable_sagemaker_autoscaling": bool_env_var("ENABLE_SM_AUTOSCALING", default=False),

    ## Example config for using alternative Tesseract OCR instead of Amazon Textract:
    "build_sagemaker_ocr": list_env_var("BUILD_SM_OCR", default=[]),
    "deploy_sagemaker_ocr": list_env_var("DEPLOY_SM_OCR", default=[]),
    "use_sagemaker_ocr": os.environ.get("USE_SM_OCR", default="tesseract"),
}

app = App()
print(f"Deploying stack with configuration:\n{json.dumps(config, indent=2)}")
demo_stack = PipelineDemoStack(
    app,
    "OCRPipelineDemo",
    **config,
)
app.synth()
