# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Masked Language Modelling dataset classes for Textract+LayoutLM

In the terms of the LayoutLM paper (https://arxiv.org/abs/1912.13318), this implementation trains a
"masked visual-language model" (predict masked token content at given position). It doesn't address
their pre-training task #2 "multi-label document classification".
"""
# Python Built-Ins:
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Dict, List, Optional, Union

# External Dependencies:
import numpy as np
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# Local Dependencies:
from ..config import DataTrainingArguments
from .base import (
    LayoutLMDataCollatorMixin,
    prepare_base_dataset,
    split_long_dataset_samples,
    TaskData,
)


logger = getLogger("data.mlm")


@dataclass
class TextractLayoutLMDataCollatorForLanguageModelling(
    LayoutLMDataCollatorMixin,
    DataCollatorForLanguageModeling,
):
    """Collator to process (batches of) Examples into batched model inputs

    For this case, tokenization can happen at the batch level which allows us to pad to the longest
    sample in batch rather than the overall model max_seq_len - for efficiency. Word splitting is
    already done by Textract, and some custom logic is required to feed through the bounding box
    inputs from Textract (at word level) to the model inputs (at token level).
    """

    def __post_init__(self):
        self._init_for_layoutlm()
        return super().__post_init__()

    def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        raise NotImplementedError(
            "Custom Textract MLM data collator has not been implemented for NumPy"
        )

    def tf_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        raise NotImplementedError(
            "Custom Textract MLM data collator has not been implemented for TensorFlow"
        )

    def torch_call(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(batch, list):
            batch = {k: [ex[k] for ex in batch] for k in batch[0]}
        else:
            batch = batch

        if self.processor:
            if "images" in batch and isinstance(batch["images"][0], list):
                # Processor needs PIL Images or np/pt tensors, but not lists:
                # (PIL->PyTorch conversion will go via numpy anyway)
                batch["images"] = [np.array(img) for img in batch["images"]]
            tokenized = self.processor(
                **{k: batch[k] for k in batch if k in self.processor_param_names},
                return_tensors=self.return_tensors,
                **self.tokenizer_extra_kwargs,
            )
        else:
            tokenized = self.tokenizer(
                **{k: batch[k] for k in batch if k in self.tokenizer_param_names},
                return_tensors=self.return_tensors,
                **self.tokenizer_extra_kwargs,
            )

        # LayoutLMV1Tokenizer also doesn't map through "boxes", but this is common across tasks so
        # it's implemented in the parent mixin:
        self._map_word_boxes(tokenized, batch["boxes"])

        # From here, implementation is as per superclass (but we can't call super because the first
        # part of the method expects batching not to have been done yet):
        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = tokenized.pop("special_tokens_mask", None)
        if self.mlm:
            tokenized["input_ids"], tokenized["labels"] = self.torch_mask_tokens(
                tokenized["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = tokenized["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            tokenized["labels"] = labels
        return tokenized


def prepare_dataset(
    textract_path: str,
    tokenizer: PreTrainedTokenizerBase,
    manifest_file_path: str,
    images_path: Optional[str] = None,
    images_prefix: str = "",
    textract_prefix: str = "",
    max_seq_len: int = 512,
    num_workers: Optional[int] = None,
):
    return split_long_dataset_samples(
        prepare_base_dataset(
            textract_path=textract_path,
            manifest_file_path=manifest_file_path,
            images_path=images_path,
            images_prefix=images_prefix,
            textract_prefix=textract_prefix,
            num_workers=num_workers,
        ),
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        num_workers=num_workers,
    )


def get_task(
    data_args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizerBase,
    processor: Optional[ProcessorMixin] = None,
    n_workers: Optional[int] = None,
) -> TaskData:
    """Load datasets and data collators for MLM model training"""
    logger.info("Getting MLM datasets")
    train_dataset = prepare_dataset(
        data_args.textract,
        tokenizer=tokenizer,
        manifest_file_path=data_args.train,
        images_path=data_args.images,
        images_prefix=data_args.images_prefix,
        textract_prefix=data_args.textract_prefix,
        max_seq_len=data_args.max_seq_length - 2,  # To allow for CLS+SEP in final
        num_workers=n_workers,
    )
    logger.info("Train dataset: %s", train_dataset)

    if data_args.validation:
        eval_dataset = prepare_dataset(
            data_args.textract,
            tokenizer=tokenizer,
            manifest_file_path=data_args.validation,
            images_path=data_args.images,
            images_prefix=data_args.images_prefix,
            textract_prefix=data_args.textract_prefix,
            max_seq_len=data_args.max_seq_length - 2,  # To allow for CLS+SEP in final
            num_workers=n_workers,
        )
    else:
        eval_dataset = None

    return TaskData(
        train_dataset=train_dataset,
        data_collator=TextractLayoutLMDataCollatorForLanguageModelling(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=data_args.pad_to_multiple_of,
            processor=processor,
        ),
        eval_dataset=eval_dataset,
        metric_computer=None,
    )
