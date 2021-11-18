# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Base/common task data utilities for Amazon Textract + LayoutLM

This module defines utilities common across the different task types (e.g. MLM, NER)
"""
# Python Built-Ins:
from dataclasses import dataclass
import json
from math import ceil
from numbers import Real
import os
import re
from typing import Callable, Dict, Generator, List, Optional, Tuple

# External Dependencies:
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction
import trp

# Local Dependencies:
from ..logging_utils import getLogger


logger = getLogger("data.base")


@dataclass
class TaskData:
    """Base data interface exposed by the different task types (MLM, NER, etc) to training scripts

    Each new task module should implement a method get_task(data_args, tokenizer) -> TaskData
    """

    train_dataset: Dataset
    data_collator: Optional[Callable] = None
    eval_dataset: Optional[Dataset] = None
    metric_computer: Optional[Callable[[EvalPrediction], Dict[str, Real]]] = None


class ExampleSplitterBase:
    """Base interface for a dataset example splitter

    In dense document processing individual pages may often be significantly longer than the
    max_seq_len of a model - rendering simple truncation of the page a poor strategy. A splitter
    defines a reproducible algorithm to split document/page text into multiple examples to stay
    within the maximum sequence length supported by the model.
    """

    @classmethod
    def n_examples(cls, n_tokens: int, max_content_seq_len: int) -> int:
        """Calculate how many individual examples are available within a given (long) text source"""
        raise NotImplementedError(
            "ExampleSplitterBase child class %s must implement n_examples()" % cls
        )

    @classmethod
    def split(
        cls,
        word_texts: List[str],
        tokenizer: PreTrainedTokenizerBase,
        max_content_seq_len: int,
    ) -> List[Tuple[int, int]]:
        """Find a set of (start, end) slices to split words for samples <= max_content_seq_len"""
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
    def split(
        cls,
        word_texts: List[str],
        tokenizer: PreTrainedTokenizerBase,
        max_content_seq_len: int,
    ) -> List[Tuple[int, int]]:
        if not (word_texts and len(word_texts)):
            return []
        tokenized = tokenizer(word_texts, add_special_tokens=False, is_split_into_words=True)
        # word_ids is List[Union[None, int]] mapping token index to word_texts index. In this case,
        # since special tokens are turned off, there are no None entries.
        word_ids = np.array(tokenized.word_ids(), dtype=int)
        n_tokens_total = len(word_ids)
        # Assuming word_ids is monotonically increasing (are there languages/tokenizers where it
        # wouldn't?), we can find the tokens which start a new word by seeing when word_ids goes up:
        token_is_new_word = np.diff(word_ids, prepend=-1)  # (1 if token is new word, 0 otherwise)
        word_start_ixs = np.squeeze(np.argwhere(token_is_new_word > 0), axis=1)
        ix_start_word = 0
        n_words = len(word_texts)
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
                    "Skipping individual 'word' which is longer than max_content_seq_len. "
                    "Something is probably wrong with your data prep. Got word '%s'"
                    % word_texts[ix_start_word]
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

        return splits


class TextractLayoutLMDatasetBase(Dataset):
    """Base class for PyTorch/Hugging Face dataset using Amazon Textract for LayoutLM-based models

    The base dataset assumes fixed/known length, which typically requires analyzing the source data
    on init - but avoids the complications of shuffling iterable dataset samples in a multi-process
    environment, or introducing SageMaker Pipe Mode and RecordIO formats.

    Source data is provided as a folder of Amazon Textract result JSONs, with an optional JSONLines
    manifest file annotating the documents in case the task is supervised.
    """

    def __init__(
        self,
        textract_path: str,
        tokenizer: PreTrainedTokenizerBase,
        manifest_file_path: Optional[str] = None,
        textract_prefix: str = "",
        max_seq_len: int = 512,
    ):
        """Initialize a TextractLayoutLMDatasetBase

        Arguments
        ---------
        textract_path : str
            The local folder where Amazon Textract result JSONs (OCR outputs) are stored.
        tokenizer : transformers.tokenization_utils_base.PreTrainedTokenizerBase
            The tokenizer for the model to be used.
        manifest_file_path : Optional[str]
            Local path to a JSON-Lines Augmented Manifest File: Optional for self-supervised
            tasks, but typically mandatory for tasks that use annotations (like entity
            recognition).
        textract_prefix : str
            s3://... URI root prefix against which the files in `textract_path` are relative.
            This is used to map `textract-ref` URIs given in the manifest file to local paths.
        max_seq_len : int
            The maximum number of tokens per sequence for the target model to be trained.
        """
        if not os.path.isdir(textract_path):
            raise ValueError("textract_path '%s' is not a valid folder" % textract_path)
        if not textract_path.endswith("/"):
            textract_path = textract_path + "/"
        self.textract_path = textract_path

        if manifest_file_path:
            if os.path.isfile(manifest_file_path):
                self.manifest_file_path = manifest_file_path
            elif os.path.isdir(manifest_file_path):
                contents = os.listdir(manifest_file_path)
                if len(contents) == 1:
                    self.manifest_file_path = os.path.join(manifest_file_path, contents[0])
                else:
                    json_contents = list(
                        filter(
                            lambda s: re.search(r"\.jsonl?$", s), map(lambda s: s.lower(), contents)
                        )
                    )
                    if len(json_contents) == 1:
                        self.manifest_file_path = os.path.join(
                            manifest_file_path,
                            json_contents[0],
                        )
                    else:
                        raise ValueError(
                            "Data manifest folder %s must contain exactly one file or exactly one "
                            ".jsonl/.json file ...Got %s" % (manifest_file_path, contents)
                        )
            else:
                raise ValueError("Data manifest '%s' is not a local file or folder")
        else:
            self.manifest_file_path = manifest_file_path

        self.textract_prefix = textract_prefix
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def textract_s3uri_to_file_path(self, s3uri: str) -> str:
        """Map a textract-ref S3 URI from manifest to local file path, via textract_prefix"""
        textract_s3key = s3uri[len("s3://") :].partition("/")[2]
        if not textract_s3key.startswith(self.textract_prefix):
            raise ValueError(
                "Textract S3 URI %s object key does not start with provided "
                "textract_prefix '%s'" % (s3uri, self.textract_prefix)
            )
        textract_relpath = textract_s3key[len(self.textract_prefix) :]
        if textract_relpath.startswith("/"):
            # Because os.path.join('anything', '/slash/prefixed') = '/slash/prefixed'
            textract_relpath = textract_relpath[1:]
        return os.path.join(self.textract_path, textract_relpath)

    def dataset_inputs(self) -> Generator[dict, None, None]:
        """Generate the sequence of manifest items with textract-ref URIs resolved locally

        Whether this dataset was instantiated with a manifest file (for annotations) or just as a
        folder of Amazon Textract JSON files, this method will yield a sequence of dicts containing
        {'textract-ref': str} resolved to the *local* path of the file, plus whatever other fields
        were present unchanged (in a manifest).
        """
        if self.manifest_file_path:
            with open(self.manifest_file_path, "r") as f:
                for linenum, line in enumerate(f, start=1):
                    logger.debug("Reading manifest line %s", linenum)
                    record = json.loads(line)
                    if "textract-ref" not in record:
                        raise ValueError(
                            f"Manifest line {linenum} missing required field 'textract-ref'"
                        )
                    else:
                        textract_ref = record["textract-ref"]
                        if textract_ref.lower().startswith("s3://"):
                            # Map S3 URI to local path:
                            textract_ref = self.textract_s3uri_to_file_path(textract_ref)
                        else:
                            # textract_fle_path in manifest isn't an S3 URI - assume rel to channel
                            if textract_ref.startswith("/"):
                                textract_ref = self.textract_path + textract_ref[1:]
                            else:
                                textract_ref = self.textract_path + textract_ref
                        # Check the resolved file path exists:
                        if not os.path.isfile(textract_ref):
                            raise ValueError(
                                "(Manifest line {}) could not find textract file {}".format(
                                    linenum,
                                    textract_ref,
                                )
                            )
                        record["textract-ref"] = textract_ref
                    yield record
        else:
            for currpath, _, files in os.walk(self.textract_path):
                for file in files:
                    yield {"textract-ref": os.path.join(currpath, file)}

    @classmethod
    def parse_textract_file(cls, file_path: str) -> trp.Document:
        """Load an Amazon Textract result JSON file via the Textract Response Parser library"""
        with open(file_path, "r") as f:
            return trp.Document(json.loads(f.read()))

    @property
    def max_content_seq_len(self):
        """Maximum content tokens per sequence after discounting required special tokens

        At this base level, datasets are assumed to have 2 special tokens: <CLS> (beginning of
        example) and <SEP> (end of example).
        """
        return self.max_seq_len - 2


@dataclass
class DummyDataCollator:
    """Data collator that just stacks tensors from inputs.

    For use with Dataset classes where the tokenization and collation leg-work is already done and
    HF's default "DataCollatorWithPadding" should explicitly *not* be used.
    """

    def __call__(self, features):
        return {k: torch.stack([f[k] for f in features]) for k in features[0]}
