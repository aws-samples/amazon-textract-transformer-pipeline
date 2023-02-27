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
import os
from typing import Dict, Optional, Union

# External Dependencies:
import datasets
import numpy as np
from transformers import PreTrainedTokenizerBase
from transformers.processing_utils import ProcessorMixin
from transformers.utils.generic import PaddingStrategy, TensorType
from transformers.tokenization_utils_base import TruncationStrategy

# Local Dependencies:
from ...config import DataTrainingArguments
from ..base import looks_like_hf_dataset, prepare_base_dataset, TaskData
from ..smgt import BBoxesWithTranscriptReviewsAnnotationResult
from ..splitting import duplicate_batch_record, remove_batch_records
from .date_normalization import generate_seq2seq_date_norm_dataset
from .metrics import get_metric_computer


logger = getLogger("data.seq2seq")


def _map_collate_seq2seq_dataset(
    batch: Dict[str, list],
    tokenizer: PreTrainedTokenizerBase,
    add_special_tokens: bool = True,
    padding: Union[bool, str, PaddingStrategy] = "max_length",
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
    """map fn to tokenize a seq2seq dataset ready for use in training"""
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


def collate_seq2seq_dataset(
    dataset: datasets.Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_input_len: Optional[int] = None,
    max_output_len: Optional[int] = None,
    pad_to_multiple_of: Optional[int] = None,
    padding: Union[bool, str, PaddingStrategy] = "max_length",
    cache_dir: Optional[str] = None,
    cache_file_prefix: Optional[str] = None,
    num_workers: Optional[int] = None,
) -> datasets.Dataset:
    """Tokenize a seq2seq dataset ready for use in training

    TODO: Should we use a DataCollator for per-batch tokenization instead?
    """
    preproc_kwargs = {
        "max_input_length": max_input_len,
        "max_output_length": max_output_len,
        "pad_to_multiple_of": pad_to_multiple_of,
        "padding": padding,
        "tokenizer": tokenizer,
    }
    return dataset.map(
        _map_collate_seq2seq_dataset,
        batched=True,
        cache_file_name=(
            os.path.join(cache_dir, f"{cache_file_prefix}_collated.arrow")
            if (cache_dir and cache_file_prefix)
            else None
        ),
        num_proc=num_workers,
        remove_columns=dataset.column_names,
        fn_kwargs=preproc_kwargs,
    )


def map_smgt_data_to_fieldnorm_seq2seq(
    batch: Dict[str, list],  # TODO: Support List[Any]? Union[Dict[List], List[Any]],
    annotation_attr: str,
):
    """Map base Textract+SMGT dataset with custom task UI inputs to a seq2seq field normalizing task

    Between the already-extracted Textract data and the boxes available on the SMGT result, you
    should have everything you need here to support validating the raw text matches the source doc
    at given locations and pulling through the source word layout boxes (similar to what we do in
    NER data prep) - but since our seq2seq models are all text-only it's not been done for now.
    """
    if annotation_attr not in batch:
        raise ValueError(f"Ground Truth label attribute '{annotation_attr}' missing from batch.")

    anns_orig = batch[annotation_attr][:]
    # Create placeholders in batch for fields to be built:
    batch["class_name"] = [None for _ in anns_orig]
    batch["src_texts"] = [None for _ in anns_orig]
    batch["tgt_texts"] = [None for _ in anns_orig]

    # Process the batch, expanding it as we go (to one record per entity):
    ix_offset = 0
    for ix_orig, ann in enumerate(anns_orig):
        ix_cur = ix_orig + ix_offset
        ann = BBoxesWithTranscriptReviewsAnnotationResult(ann)
        valid_entities = [
            ent
            for ent in ann.entities
            if ent.label is not None and ent.raw_text is not None and ent.target_text is not None
        ]
        n_valid_entities = len(valid_entities)
        if n_valid_entities == 0:
            batch = remove_batch_records(batch, ix_cur, n=1)
            ix_offset -= 1
        else:
            batch = duplicate_batch_record(
                batch,
                ix_cur,
                n_valid_entities,
                feature_overrides={
                    "class_name": [ent.label for ent in valid_entities],
                    "src_texts": [
                        f"Normalize {ent.label}: {ent.raw_text}" for ent in valid_entities
                    ],
                    "tgt_texts": [ent.target_text for ent in valid_entities],
                },
            )
            ix_offset += n_valid_entities - 1

    return batch


def prepare_dataset(
    data_path: str,
    annotation_attr: Optional[str] = None,
    textract_path: Optional[str] = None,
    images_path: Optional[str] = None,
    images_prefix: str = "",
    textract_prefix: str = "",
    num_workers: Optional[int] = None,
    batch_size: int = 16,
    cache_dir: Optional[str] = None,
    cache_file_prefix: Optional[str] = None,
) -> datasets.Dataset:

    if looks_like_hf_dataset(data_path):
        # Pre-prepared dataset, just load and return:
        return datasets.load_from_disk(data_path)

    # Else we need to prepare the dataset from Textract/SMGT files.
    dataset = prepare_base_dataset(
        textract_path=textract_path,
        manifest_file_path=data_path,
        images_path=images_path,
        images_prefix=images_prefix,
        textract_prefix=textract_prefix,
        num_workers=num_workers,
        batch_size=batch_size,
        cache_dir=cache_dir,
        map_cache_file_name=(
            os.path.join(cache_dir, f"{cache_file_prefix}_1base.arrow")
            if (cache_dir and cache_file_prefix)
            else None
        ),
    ).map(
        map_smgt_data_to_fieldnorm_seq2seq,
        batched=True,
        batch_size=batch_size,
        fn_kwargs={"annotation_attr": annotation_attr},
        num_proc=num_workers,
        desc="Extracting seq2seq examples from Ground Truth annotations",
        cache_file_name=(
            os.path.join(cache_dir, f"{cache_file_prefix}_2label.arrow")
            if (cache_dir and cache_file_prefix)
            else None
        ),
    )

    # Since this is a field-text normalization task, splitting long samples is not supported (e.g.
    # with `split_long_dataset_samples()` as in other tasks)
    return dataset


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
        train_dataset = prepare_dataset(
            data_path=data_args.train,
            annotation_attr=data_args.annotation_attr,
            textract_path=data_args.textract,
            images_path=data_args.images,
            images_prefix=data_args.images_prefix,
            textract_prefix=data_args.textract_prefix,
            num_workers=n_workers,
            batch_size=data_args.dataproc_batch_size,
            cache_dir=cache_dir,
            cache_file_prefix="seq2seqtrain",
        )
        logger.info("Train dataset ready: %s", train_dataset)
    else:
        # TODO: Factor generation+preprocessing into separate generate_dataset fn?
        logger.info("Generating new synthetic seq2seq training dataset")
        train_dataset = generate_seq2seq_date_norm_dataset(
            n=1000,
            rng=rng,
        )

    train_dataset = collate_seq2seq_dataset(
        train_dataset,
        tokenizer,
        max_input_len=data_args.max_seq_length - 2,
        max_output_len=64,  # TODO: Parameterize?
        pad_to_multiple_of=data_args.pad_to_multiple_of,
        cache_dir=cache_dir,
        cache_file_prefix="seq2seqtrain",
    )

    if data_args.validation:
        eval_dataset = prepare_dataset(
            data_path=data_args.validation,
            annotation_attr=data_args.annotation_attr,
            textract_path=data_args.textract,
            images_path=data_args.images,
            images_prefix=data_args.images_prefix,
            textract_prefix=data_args.textract_prefix,
            num_workers=n_workers,
            batch_size=data_args.dataproc_batch_size,
            cache_dir=cache_dir,
            cache_file_prefix="seq2seqval",
        )
        logger.info("Validation dataset ready: %s", eval_dataset)
    else:
        if not data_args.train:
            logger.info("Generating new synthetic seq2seq validation dataset")
            eval_dataset = generate_seq2seq_date_norm_dataset(n=200, rng=rng)
        else:
            # Can't assume it's the date norm task: Leave no val set
            eval_dataset = None

    if eval_dataset:
        eval_dataset = collate_seq2seq_dataset(
            eval_dataset,
            tokenizer,
            max_input_len=data_args.max_seq_length - 2,
            max_output_len=128,  # TODO: Parameterize?
            pad_to_multiple_of=data_args.pad_to_multiple_of,
            cache_dir=cache_dir,
            cache_file_prefix="seq2seqval",
        )

    return TaskData(
        train_dataset=train_dataset,
        data_collator=None,
        eval_dataset=eval_dataset,
        metric_computer=get_metric_computer(tokenizer),
    )
