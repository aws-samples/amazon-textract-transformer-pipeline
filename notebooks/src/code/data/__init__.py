# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Data loading utilities for Amazon Textract with Hugging Face Transformers

Call get_datasets() from the training script to load datasets/collators for the current task.
"""
# Python Built-Ins:
from typing import Iterable, Optional

# External Dependencies:
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# Local Dependencies:
from ..config import DataTrainingArguments
from .base import TaskData
from .mlm import get_task as get_mlm_task
from .ner import get_task as get_ner_task
from .seq2seq import get_task as get_seq2seq_task


def get_datasets(
    data_args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizerBase,
    processor: Optional[ProcessorMixin] = None,
    model_param_names: Optional[Iterable[str]] = None,
    n_workers: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> TaskData:
    """Load datasets and data collators for model pre/training"""
    if data_args.task_name == "mlm":
        return get_mlm_task(
            data_args,
            tokenizer,
            processor,
            model_param_names=model_param_names,
            n_workers=n_workers,
            cache_dir=cache_dir,
        )
    elif data_args.task_name == "ner":
        return get_ner_task(
            data_args, tokenizer, processor, n_workers=n_workers, cache_dir=cache_dir
        )
    elif data_args.task_name == "seq2seq":
        return get_seq2seq_task(
            data_args, tokenizer, processor, n_workers=n_workers, cache_dir=cache_dir
        )
    else:
        raise ValueError(
            "Unknown task '%s' is not in 'mlm', 'ner', 'seq2seq'" % data_args.task_name
        )
