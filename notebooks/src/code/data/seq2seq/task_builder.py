# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Main 'task' builder for seq2seq tasks used by the model training script

Collects seq2seq data (e.g. date normalization) into overall task format expected by the training
script (matching other tasks like NER, MLM).

One interesting aspect of the conditional prompting framing of this seq2seq task, is that you could
train a single model to perform multiple kinds of field normalization by using different prompts.
For example "Convert date ...: ..." vs "Normalize currency ...: ..." and so on.

Here we just show a single-task date normalizing example.
"""
# Python Built-Ins:
from logging import getLogger
from numbers import Real
import os
from typing import Callable, Dict, Optional, Union

# External Dependencies:
import datasets
import numpy as np
from transformers import EvalPrediction, PreTrainedTokenizerBase
from transformers.processing_utils import ProcessorMixin
from transformers.utils.generic import PaddingStrategy, TensorType
from transformers.tokenization_utils_base import TruncationStrategy

# Local Dependencies:
from ...config import DataTrainingArguments
from ..base import TaskData
from .date_normalization import generate_seq2seq_date_norm_dataset


logger = getLogger("data.seq2seq")


def _preprocess_seq2seq_dataset(
    batch: Dict[str, list],
    tokenizer: PreTrainedTokenizerBase,
    add_special_tokens: bool = True,
    padding: Union[bool, str, PaddingStrategy] = False,
    truncation: Union[bool, str, TruncationStrategy] = None,
    max_input_length: Optional[int] = None,
    max_output_length: Optional[int] = None,
    stride: int = 0,
    is_split_into_words: bool = False,
    pad_to_multiple_of: Optional[int] = None,
    return_tensors: Union[str, TensorType, None] = None,
    return_token_type_ids: Optional[bool] = None,
    return_attention_mask: Optional[bool] = None,
    return_overflowing_tokens: bool = False,
    return_special_tokens_mask: bool = False,
    return_offsets_mapping: bool = False,
    return_length: bool = False,
    verbose: bool = True,
) -> Dict[str, list]:
    """map fn to tokenize a seq2seq dataset ready for use in training

    TODO: Should we use a DataCollator for per-batch tokenization instead?
    """
    # encode the documents
    prompts = batch["src_texts"]
    answers = batch["tgt_texts"]

    # Encode the inputs:
    model_inputs = tokenizer(
        prompts,
        add_special_tokens=add_special_tokens,
        padding=padding,
        truncation=truncation,
        max_length=max_input_length,
        stride=stride,
        is_split_into_words=is_split_into_words,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors=return_tensors,
        return_token_type_ids=return_token_type_ids,
        return_attention_mask=return_attention_mask,
        return_overflowing_tokens=return_overflowing_tokens,
        return_special_tokens_mask=return_special_tokens_mask,
        return_offsets_mapping=return_offsets_mapping,
        return_length=return_length,
        verbose=verbose,
    )

    # Encode the targets:
    labels = tokenizer(
        answers,
        add_special_tokens=add_special_tokens,
        padding=padding,
        truncation=truncation,
        max_length=max_output_length,
        stride=stride,
        is_split_into_words=is_split_into_words,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors=return_tensors,
        return_token_type_ids=return_token_type_ids,
        return_attention_mask=return_attention_mask,
        return_overflowing_tokens=return_overflowing_tokens,
        return_special_tokens_mask=return_special_tokens_mask,
        return_offsets_mapping=return_offsets_mapping,
        return_length=return_length,
        verbose=verbose,
    ).input_ids

    # important: we need to replace the index of the padding tokens by -100
    # such that they are not taken into account by the CrossEntropyLoss
    labels_with_ignore_index = []
    for labels_example in labels:
        labels_example = [label if label != 0 else -100 for label in labels_example]
        labels_with_ignore_index.append(labels_example)

    model_inputs["labels"] = labels_with_ignore_index
    return model_inputs


def get_metric_computer(
    tokenizer: PreTrainedTokenizerBase,
) -> Callable[[EvalPrediction], Dict[str, Real]]:
    """An 'accuracy' computer for seq2seq tasks that ignores outer whitespace and case.

    For our example task, it's reasonable to measure exact-match accuracy (since we're normalising
    small text spans - not e.g. summarizing long texts to shorter paragraphs). Therefore this metric
    computer checks exact accuracy, while allowing for variations in case and leading/trailing
    whitespace.
    """

    def compute_metrics(p: EvalPrediction) -> Dict[str, Real]:
        # Convert model output probs/logits to predicted token IDs:
        predicted_token_ids = np.argmax(p.predictions[0], axis=2)
        # Replace everything from the first <end-of-sentence> token onward with padding (as eos
        # would terminate generation in a normal generate() call)
        for ix_batch, seq in enumerate(predicted_token_ids):
            eos_token_matches = np.where(seq == tokenizer.eos_token_id)
            if len(eos_token_matches) and len(eos_token_matches[0]):
                first_eos_posn = eos_token_matches[0][0]
                predicted_token_ids[ix_batch, first_eos_posn:] = tokenizer.pad_token_id

        gen_texts = [
            s.strip().lower()
            for s in tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)
        ]

        target_texts = [
            s.strip().lower()
            for s in tokenizer.batch_decode(
                # Replace label '-100' tokens (ignore index for BinaryCrossEntropy) with '0' (<pad>
                # token), to avoid an OverflowError when trying to decode the target text:
                np.maximum(0, p.label_ids),
                skip_special_tokens=True,
            )
        ]

        n_examples = len(gen_texts)
        n_correct = sum(1 for gen, target in zip(gen_texts, target_texts) if gen == target)
        return {
            "n_examples": len(gen_texts),
            "acc": n_correct / n_examples,
        }

    return compute_metrics


def get_task(
    data_args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizerBase,
    processor: Optional[ProcessorMixin] = None,
    # model_param_names: Optional[Iterable[str]] = None,
    n_workers: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> TaskData:
    """Load datasets and data collators for seq2seq model training"""
    logger.info("Getting seq2seq datasets")

    # TODO: Currently non-reproducible, but we don't have access to CLI arg seed here
    # Best practice for now would be to generate your dataset before running training anyway,
    # instead of relying on ephemeral dataset generation within the job.
    rng = np.random.default_rng()

    # Load or create the training and validation datasets:
    if data_args.train:
        logger.info("Loading seq2seq training dataset from disk %s", data_args.train)
        train_dataset = datasets.load_from_disk(data_args.train)
    else:
        logger.info("Generating new synthetic seq2seq training dataset")
        train_dataset = generate_seq2seq_date_norm_dataset(n=1000, rng=rng)

    if data_args.validation:
        logger.info("Loading seq2seq validation dataset from disk %s", data_args.validation)
        eval_dataset = datasets.load_from_disk(data_args.validation)
    else:
        logger.info("Generating new synthetic seq2seq validation dataset")
        eval_dataset = generate_seq2seq_date_norm_dataset(n=200, rng=rng)

    # Pre-process the datasets with the tokenizer:
    preproc_kwargs = {
        "max_input_length": data_args.max_seq_length - 2,  # To allow for CLS+SEP in final
        "max_output_length": 64,  # TODO: Parameterize?
        "pad_to_multiple_of": data_args.pad_to_multiple_of,
        "padding": "max_length",
        "tokenizer": tokenizer,
    }
    train_dataset = train_dataset.map(
        _preprocess_seq2seq_dataset,
        batched=True,
        cache_file_name=(os.path.join(cache_dir, "seq2seqtrain.arrow") if cache_dir else None),
        num_proc=n_workers,
        remove_columns=train_dataset.column_names,
        fn_kwargs=preproc_kwargs,
    )
    eval_dataset = eval_dataset.map(
        _preprocess_seq2seq_dataset,
        batched=True,
        cache_file_name=(os.path.join(cache_dir, "seq2seqeval.arrow") if cache_dir else None),
        num_proc=n_workers,
        remove_columns=eval_dataset.column_names,
        fn_kwargs=preproc_kwargs,
    )

    return TaskData(
        train_dataset=train_dataset,
        data_collator=None,
        eval_dataset=eval_dataset,
        metric_computer=get_metric_computer(tokenizer),
    )
