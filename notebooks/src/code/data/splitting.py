# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Utilities for sensibly splitting long dataset examples for max_seq_len-limited models
"""
# Python Built-Ins:
from math import ceil
from typing import Any, Dict, Iterable, List, Set, Tuple, Type

# External Dependencies:
import numpy as np
from transformers import BatchEncoding
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# Local Dependencies:
from ..logging_utils import getLogger


logger = getLogger("data.splitting")


class ExampleSplitterBase:
    """Base interface for a dataset example splitter

    In dense document processing individual pages may often be significantly longer than the
    max_seq_len of a model - rendering simple truncation of the page a poor strategy. A splitter
    defines a reproducible algorithm to split document/page text into multiple examples, to stay
    within the maximum sequence length supported by the model.
    """

    @classmethod
    def n_examples(cls, n_tokens: int, max_content_seq_len: int) -> int:
        """Calculate how many individual examples are available within a given (long) text source"""
        raise NotImplementedError(
            "ExampleSplitterBase child class %s must implement n_examples()" % cls
        )

    @classmethod
    def batched_split(
        cls,
        tokenized: BatchEncoding,
        n_words_by_sample: List[int],
        max_content_seq_len: int,
    ) -> List[Tuple[int, int]]:
        """Find sets of (start, end) slices to split example words for max_content_seq_len tokens

        This function takes a tokenized data batch (transformers.BatchEncoding) and returns *word*
        level splits for each sample in the batch, to keep the resultant *tokens* under
        max_content_seq_len for each sample.

        Parameters
        ----------
        tokenized :
            An already-tokenized (with no truncation or padding) encoding of the input data
        n_words_by_sample :
            Pre-calculated number of words per example in the batch
        max_content_seq_len :
            Maximum number of tokens per sample to split to
        """
        # How do we split a tokenized? What makes sense to return?
        raise NotImplementedError("ExampleSplitterBase child class %s must implement split()" % cls)


class NaiveExampleSplitter(ExampleSplitterBase):
    """Split sequences by word, and pull final sequence start forward if it comes up <50% max len

    This algorithm produces examples by splitting tokens on word boundaries, extending each sample
    until max_content_seq_len is filled. *IF* the final generated example is less than 50% of the
    maximum tokens, its start index will be pulled forward to consume as many words as will fit.
    Apart from this, there will be no overlap between examples.
    """

    @classmethod
    def n_examples(cls, n_tokens: int, max_content_seq_len: int) -> int:
        return int(ceil(n_tokens / max_content_seq_len))

    @classmethod
    def batched_split(
        cls,
        tokenized: BatchEncoding,
        n_words_by_sample: List[int],
        max_content_seq_len: int,
    ) -> List[Tuple[int, int]]:
        tokenized_lens = [len(input) for input in tokenized["input_ids"]]
        splits_by_sample = []

        for ixsample, n_tokens_total in enumerate(tokenized_lens):
            n_words = n_words_by_sample[ixsample]

            # word_ids is List[Union[None, int]] mapping token index to word_texts index. In this
            # case, since special tokens are turned off, there are no None entries.
            word_ids = np.array(tokenized.word_ids(ixsample), dtype=int)

            # Assuming word_ids is monotonically increasing (are there languages/tokenizers where
            # it wouldn't?), we can find the tokens which start a new word by seeing when word_ids
            # goes up:
            token_is_new_word = np.diff(word_ids, prepend=-1)  # (1 if token is new word, else 0)
            word_start_ixs = np.squeeze(np.argwhere(token_is_new_word > 0), axis=1)

            ix_start_word = 0
            splits = []
            while ix_start_word < n_words:
                start_token = word_start_ixs[ix_start_word]
                end_token = start_token
                ix_end_word = ix_start_word
                # Seek forward to include as many words as fit:
                while ix_end_word < n_words:
                    next_ix_end_word = ix_end_word + 1
                    next_end_token = (
                        word_start_ixs[next_ix_end_word]
                        if next_ix_end_word < n_words
                        else n_tokens_total
                    )
                    if next_end_token - start_token > max_content_seq_len:
                        break
                    else:
                        ix_end_word = next_ix_end_word
                        end_token = next_end_token
                # Extreme edge case:
                # If the current word was longer than max_content_seq_len by itself, we need to skip it
                # to avoid an infinite loop
                if end_token == start_token:
                    logger.warning(
                        # TODO: Is there anything in tokenized to let us report what the word is here?
                        "Skipping individual 'word' which is longer than max_content_seq_len. "
                        "Something is probably wrong with your data prep."
                    )
                    ix_start_word += 1
                    continue
                # If the resultant sample is short, also seek backward to add extra context:
                if end_token - start_token < max_content_seq_len * 0.5:
                    while ix_start_word > 0:
                        next_ix_start_word = ix_start_word - 1
                        next_start_token = word_start_ixs[next_ix_start_word]
                        if end_token - next_start_token > max_content_seq_len:
                            break
                        else:
                            ix_start_word = next_ix_start_word
                            start_token = next_start_token
                # Log the split and move on to find the next one
                splits.append((ix_start_word, ix_end_word))
                ix_start_word = ix_end_word
            splits_by_sample.append(splits)

        return splits_by_sample


def duplicate_batch_record(
    batch: Dict[str, List[Any]],
    ix: int,
    n_copies: int,
    feature_overrides: Dict[str, List[Any]],
) -> Dict[str, List[Any]]:
    """Copy one record in a batch encoding

    Parameters
    ----------
    batch :
        Input data, dictionary by feature name of value lists.
    ix :
        0-based index of record to expand
    n_copies :
        Target number of copies of the record in the output (1 = no change, 0 = remove the record)
    feature_overrides :
        Dictionary by feature name of value lists to set. For all features not in this dict, the
        record's input value will be duplicated n_copies times. This function assumes but does not
        validate that all values in feature_overrides have length n_copies.

    Returns
    -------
    result :
        A shallow copy of the batch with the target record copied `n_copies` times.
    """
    return {
        name: (
            values[:ix]
            + (feature_overrides[name] if name in feature_overrides else [values[ix]] * n_copies)
            + values[ix + 1 :]
        )
        for name, values in batch.items()
    }


def remove_batch_records(
    batch: Dict[str, List[Any]],
    ix_start: int,
    n: int = 1,
) -> Dict[str, List[Any]]:
    """Remove one or more records from a batch

    Parameters
    ----------
    batch :
        Input data, dictionary by feature name of value lists.
    ix_start :
        0-based index of first record to remove
    n :
        Number of records to remove

    Returns
    -------
    result :
        A shallow copy of the batch with the target record(s) removed.
    """
    return {name: (values[:ix_start] + values[ix_start + n :]) for name, values in batch.items()}


def split_batch_record(
    batch: Dict[str, List[Any]],
    ix: int,
    splits: List[Tuple[int, int]],
    exclude_features: Iterable[str],
) -> Dict[str, List[Any]]:
    """Split one record in a batch into multiple

    Parameters
    ----------
    batch :
        Input data, dictionary by feature name of value lists.
    ix :
        0-based index of record to split
    splits :
        List of (start, end) tuples - each defining a split on the record.
    exclude_features :
        For all batch features *not* in this list, if the feature's value for the particular record
        `ix` is itself a list, then the list will be partitioned by the given splits. If the value
        is not list-like or is in `exclude_features`, then it will be replicated across all new
        records. It's okay to include names here that aren't in `batch`.

    Returns
    -------
    result :
        A shallow copy of the batch with the target record split by `splits`.
    """

    def is_list_like(x):
        return hasattr(x, "__getitem__") and not isinstance(x, (str, dict))

    return {
        name: (
            values[:ix]
            + (
                [values[ix]] * len(splits)
                if (name in exclude_features or not is_list_like(values[ix]))
                else [values[ix][slice(*split)] for split in splits]
            )
            + values[ix + 1 :]
        )
        for name, values in batch.items()
    }


def map_split_long_samples(
    # TODO: Should we also support List[Dict[str, Any]] here?
    batch: Dict[str, List],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int = 512,
    tokenizer_params: Set[str] = set(),
    splitter: Type[ExampleSplitterBase] = NaiveExampleSplitter,
) -> Dict[str, List]:
    """datasets.map function to split long examples down for model max_seq_len

    Parameters
    ----------
    batch :
        Dataset batch to process. Must include "text". Depending on the tokenizer, may also need to
        include "images", "boxes".
    tokenizer :
        Tokenizer to apply to calculate raw sample lengths
    max_seq_len :
        Maximum number of text tokens per sample after splitting. **NOTE:** If your practical task
        will add special tokens such as CLS, SEP, you'll need to subtract these from your model's
        overall maximum sequence length.
    tokenizer_params :
        `set` of parameter names accepted by the tokenizer, pre-calculated for performance
        rather than inspecting the tokenizer's signature for every processed batch. This is used to
        match configuration between different models' tokenizers (e.g. LayoutLMV1 vs v2).
    splitter :
        A subtype of `ExampleSplitterBase` to use to evaluate and split the sequences.

    Returns
    -------
    batch :
        Batch with long sequences split into multiple examples (so output batch may have more
        examples than input).
    """
    tokenizer_kwargs = {"add_special_tokens": False}
    if "is_split_into_words" in tokenizer_params:
        tokenizer_kwargs["is_split_into_words"] = True  # (for LayoutLMv1)
    if "boxes" in tokenizer_params:
        tokenizer_kwargs["boxes"] = batch["boxes"]

    tokenized = tokenizer(batch["text"], **tokenizer_kwargs)
    batch_splits = splitter.batched_split(
        tokenized,
        [len(texts) for texts in batch["text"]],
        max_content_seq_len=max_seq_len,
    )

    # Ensure that even if no samples need splitting, we return a shallow copy rather than the
    # original batch object (not doing this was observed to cause problems in datasets 2.2.1):
    batch = {k: [val for val in v] for k, v in batch.items()}

    input_n_records = len(batch["text"])
    input_to_output = list(range(input_n_records))
    for iin, splits in enumerate(batch_splits):
        n_record_splits = len(splits)
        if n_record_splits > 1:
            ix = input_to_output[iin]
            logger.debug(
                "Splitting original record %s (starting index %s) into %s splits: %s",
                iin,
                ix,
                n_record_splits,
                splits,
            )
            batch = split_batch_record(batch, ix, splits, exclude_features=set(("images",)))
            input_to_output = input_to_output[0 : iin + 1] + [
                ix + n_record_splits - 1 for ix in input_to_output[iin + 1 :]
            ]

    return batch
