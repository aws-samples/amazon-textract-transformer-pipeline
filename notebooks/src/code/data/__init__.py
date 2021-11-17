# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Data loading utilities for Amazon Textract with Hugging Face Transformers

Call get_datasets() from the training script to load datasets/collators for the current task.
"""
# External Dependencies:
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# Local Dependencies:
from ..config import DataTrainingArguments
from .base import TaskData
from .mlm import get_task as get_mlm_task
from .ner import get_task as get_ner_task


def get_datasets(
    data_args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizerBase,
) -> TaskData:
    """Load datasets and data collators for model pre/training"""
    if data_args.task_name == "mlm":
        return get_mlm_task(data_args, tokenizer)
    elif data_args.task_name == "ner":
        return get_ner_task(data_args, tokenizer)
    else:
        raise ValueError("Unknown task '%s' is not 'mlm' or 'ner'" % data_args.task_name)
