# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Unique ID utilities for SageMaker"""

from datetime import datetime


def append_timestamp(s: str, sep: str = "-", include_millis=True) -> str:
    """Append current datetime to `s` in a format suitable for SageMaker job names"""
    now = datetime.now()
    if include_millis:
        # strftime only supports microseconds, so we trim by 3:
        datetime_str = now.strftime(f"%Y{sep}%m{sep}%d{sep}%H{sep}%M{sep}%S{sep}%f")[:-3]
    else:
        datetime_str = now.strftime(f"%Y{sep}%m{sep}%d{sep}%H{sep}%M{sep}%S")
    return sep.join((s, datetime_str))
