# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Shared utilities for the pipeline CDK app"""
# Python Built-Ins:
import os
from typing import Union


def abs_path(rel_path: Union[str, os.PathLike], from__file__: str) -> str:
    """Construct an absolute path from a relative path and current __file__ location"""
    return os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.realpath(from__file__)),
            rel_path,
        )
    )
