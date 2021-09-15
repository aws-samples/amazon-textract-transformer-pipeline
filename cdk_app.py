#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""AWS CDK app entry point for OCR pipeline sample
"""

# External Dependencies:
from aws_cdk import core as cdk

# Local Dependencies:
from annotation import AnnotationInfraStack
from pipeline import ProcessingPipelineStack


app = cdk.App()
annotation_stack = AnnotationInfraStack(app, "AnnotationStack")
pipeline_stack = ProcessingPipelineStack(app, "PipelineStack")
app.synth()
