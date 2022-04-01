#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""AWS CDK app entry point for OCR pipeline sample
"""

# External Dependencies:
from aws_cdk import App

# Local Dependencies:
from cdk_demo_stack import PipelineDemoStack


app = App()
demo_stack = PipelineDemoStack(app, "OCRPipelineDemo", default_project_id="ocr-transformers-demo")
app.synth()
