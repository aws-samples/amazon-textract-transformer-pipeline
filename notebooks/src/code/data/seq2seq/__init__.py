# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Data utilities for generative, sequence-to-sequence tasks

This task is experimental, and does not currently support layout-aware models. As shown in the
'Optional Extras' notebook, you can use it to train separate post-processing models to normalize
extracted fields: For example converting the format of dates.
"""
from .task_builder import get_task
