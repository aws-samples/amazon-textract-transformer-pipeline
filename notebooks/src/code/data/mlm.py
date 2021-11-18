# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Masked Language Modelling dataset classes for Textract+LayoutLM

In the terms of the LayoutLM paper (https://arxiv.org/abs/1912.13318), this implementation trains a
"masked visual-language model" (predict masked token content at given position). It doesn't address
their pre-training task #2 "multi-label document classification".
"""
# Python Built-Ins:
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, Union

# External Dependencies:
import numpy as np
import torch
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# Local Dependencies:
from ..config import DataTrainingArguments
from .base import ExampleSplitterBase, NaiveExampleSplitter, TaskData, TextractLayoutLMDatasetBase
from .geometry import layoutlm_boxes_from_trp_blocks


@dataclass
class TextractLayoutLMExampleForLM:
    """Data class yielded as examples by the MLM dataset for subsequent collation and training"""

    word_boxes_normalized: np.ndarray  # Nx4 array already normalized to LayoutLM 0-1000 format
    word_texts: List[str]  # List of length N, text per Textract WORD block


@dataclass
class TextractLayoutLMDataCollatorForLanguageModelling(DataCollatorForLanguageModeling):
    """Collator to process (batches of) Examples into batched model inputs

    For this case, tokenization can happen at the batch level which allows us to pad to the longest
    sample in batch rather than the overall model max_seq_len - for efficiency. Word splitting is
    already done by Textract, and some custom logic is required to feed through the bounding box
    inputs from Textract (at word level) to the model inputs (at token level).
    """

    bos_token_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    pad_token_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    sep_token_box: Tuple[int, int, int, int] = (1000, 1000, 1000, 1000)

    def __post_init__(self):
        self._special_token_boxes = torch.LongTensor(
            [
                self.bos_token_box,
                self.pad_token_box,
                self.sep_token_box,
            ]
        )
        return super().__post_init__()

    def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        raise NotImplementedError(
            "Custom Textract MLM data collator has not been implemented for NumPy"
        )

    def tf_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        raise NotImplementedError(
            "Custom Textract MLM data collator has not been implemented for TensorFlow"
        )

    def torch_call(self, examples: List[TextractLayoutLMExampleForLM]) -> Dict[str, Any]:
        # Tokenize, pad and etc the words:
        batch = self.tokenizer(
            [example.word_texts for example in examples],
            is_split_into_words=True,
            return_attention_mask=True,
            padding=bool(self.pad_to_multiple_of),
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        # Map through the bounding boxes to the generated tokens:
        # We do this by augmenting the list of word bboxes to include the special token bboxes,
        # editing the word_ids mapping from tokens->words to match special tokens to their special
        # boxes (instead of None), and then applying this set of indexes to produce the token-wise
        # boxes including special tokens.
        bbox_tensors_by_example = []
        for ixex in range(len(examples)):
            box_ids = batch.word_ids(ixex)  # List[Union[int, None]], =None at special tokens
            n_real_boxes = len(examples[ixex].word_boxes_normalized)
            augmented_example_word_boxes = torch.cat(
                (
                    torch.LongTensor(examples[ixex].word_boxes_normalized),
                    self._special_token_boxes,
                ),
                dim=0,
            )
            if box_ids[0] is None:  # Shortcut as <bos> should appear only at start
                box_ids[0] = n_real_boxes  # bos_token_box, per _special_token_boxes
            # Torch tensors don't support None, but numpy float ndarrays do:
            box_ids_np = np.array(box_ids, dtype=float)
            box_ids_np = np.where(
                batch.input_ids[ixex, :] == self.tokenizer.pad_token_id,
                n_real_boxes + 1,  # pad_token_box, per _special_token_boxes
                box_ids_np,
            )
            box_ids_np = np.where(
                batch.input_ids[ixex, :] == self.tokenizer.sep_token_id,
                n_real_boxes + 2,  # sep_token_box, per _special_token_boxes
                box_ids_np,
            )
            bbox_tensors_by_example.append(
                torch.index_select(
                    augmented_example_word_boxes,
                    0,
                    # By this point all NaNs from special tokens should be resolved so can cast:
                    torch.LongTensor(box_ids_np.astype(int)),
                )
            )
        batch["bbox"] = torch.stack(bbox_tensors_by_example)

        # From here, implementation is as per superclass (but we can't call super because the first
        # part of the method expects batching not to have been done yet):
        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch


class TextractLayoutLMDatasetForLM(TextractLayoutLMDatasetBase):
    """PyTorch/Hugging Face dataset for masked language modelling with LayoutLM & Amazon Textract

    This implementation parses the dataset up-front to calculate an overall length (in number of
    examples, after splitting any long pages) and enable random access - rather than having to
    worry about implementing order randomization in a streaming API.

    We assume the dataset (as represented in example_index) fits into available memory for
    simplicity and performance. If that's not the case, you could consider other options - for
    example serializing example_index out to disk and reloading entries on-demand in __getitem__.
    """

    splitter: ClassVar[Type[ExampleSplitterBase]] = NaiveExampleSplitter

    def __init__(
        self,
        textract_path: str,
        tokenizer: PreTrainedTokenizerBase,
        manifest_file_path: Optional[str] = None,
        textract_prefix: str = "",
        max_seq_len: int = 512,
    ):
        """Initialize a TextractLayoutLMDatasetForLM"""
        super().__init__(
            textract_path,
            tokenizer,
            manifest_file_path=manifest_file_path,
            textract_prefix=textract_prefix,
            max_seq_len=max_seq_len,
        )
        self.example_index: List[TextractLayoutLMExampleForLM] = []

        # Load the raw manifest data:
        for record in self.dataset_inputs():
            textract_file_path = record["textract-ref"]
            page_num = record.get("page-num")
            doc = self.parse_textract_file(textract_file_path)
            for page in doc.pages[
                # Filter to target page if provided, else process all pages:
                slice(None)
                if page_num is None
                else slice(page_num - 1, page_num)
            ]:
                words = [word for line in page.lines for word in line.words]
                word_boxes = layoutlm_boxes_from_trp_blocks(words)
                word_texts = [word.text for word in words]
                for start_word, end_word in self.splitter.split(
                    word_texts,
                    tokenizer,
                    self.max_content_seq_len,
                ):
                    self.example_index.append(
                        TextractLayoutLMExampleForLM(
                            word_boxes_normalized=word_boxes[start_word:end_word, :],
                            word_texts=word_texts[start_word:end_word],
                        )
                    )

    def __getitem__(self, idx: int):
        return self.example_index[idx]

    def __len__(self) -> int:
        return len(self.example_index)


def get_task(
    data_args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizerBase,
) -> TaskData:
    """Load datasets and data collators for Masked Language Modelling"""
    train_dataset = TextractLayoutLMDatasetForLM(
        textract_path=data_args.textract,
        tokenizer=tokenizer,
        max_seq_len=data_args.max_seq_length,
        manifest_file_path=data_args.train,
        textract_prefix=data_args.textract_prefix,
    )

    return TaskData(
        train_dataset=train_dataset,
        data_collator=TextractLayoutLMDataCollatorForLanguageModelling(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=8,
        ),
        eval_dataset=TextractLayoutLMDatasetForLM(
            textract_path=data_args.textract,
            tokenizer=tokenizer,
            max_seq_len=data_args.max_seq_length,
            manifest_file_path=data_args.validation,
            textract_prefix=data_args.textract_prefix,
        )
        if data_args.validation
        else None,
        metric_computer=None,
    )
