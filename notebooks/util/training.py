# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Utilities to support the model training on SageMaker"""


def get_hf_metric_regex(metric_name: str) -> str:
    """Build RegEx string to extract a numeric HuggingFace Transformers metric from SageMaker logs

    HF metric log lines look like a Python dict print e.g:
    {'eval_loss': 0.3940396010875702, ..., 'epoch': 1.0}
    """
    scientific_number_exp = r"(-?[0-9]+(\.[0-9]+)?(e[+\-][0-9]+)?)"
    return "".join(
        (
            "'",
            metric_name,
            "': ",
            scientific_number_exp,
            "[,}]",
        )
    )
