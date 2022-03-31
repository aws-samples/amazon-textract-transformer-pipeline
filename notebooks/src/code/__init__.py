# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Amazon Textract + LayoutLM model training and inference code package for SageMaker

Why the extra level of nesting? Because the src folder (even if __init__ is present) is not loaded
as a Python module during training, but rather as the working directory. This requires a different
import syntax for top-level files/folders (`import config`, not `from . import config`) than you
would see if your working directory was different (for example when you `from src import code` to
use it from one of the notebooks).

Wrapping this code in an extra package folder ensures that - regardless of whether you use it from
notebook, in SM training job, or in some other app - relative imports *within* this code/ folder
work correctly.
"""
