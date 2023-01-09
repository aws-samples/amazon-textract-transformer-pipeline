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
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# External Dependencies:
import numpy as np
import torch
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

    model_param_names: Optional[Iterable[str]] = None
    tim_probability: float = 0.2  # 15% replaced plus 5% dropped, per LMv2 paper (drop not impl.)
    tiam_probability: float = 0.15  # 15% covered, per LMv2 paper

    def __post_init__(self):
        self._init_for_layoutlm()

        # Configuration diagnostics:
        if self.tim_probability > 0:
            if self.model_param_names is None or "image_mask_label" not in self.model_param_names:
                logger.warning(
                    "model_param_names does not contain image_mask_label: Ignoring configured "
                    "tim_probability. Text-Image Matching will be disabled."
                )
        elif self.model_param_names is not None and ("image_mask_label" in self.model_param_names):
            logger.warning("Pre-training with Text-Image Matching disabled (tim_probability = 0)")
        if self.tiam_probability > 0:
            if self.model_param_names is None or "imline_mask_label" not in self.model_param_names:
                logger.warning(
                    "model_param_names does not contain imline_mask_label: Ignoring configured "
                    "tiam_probability. Text-Image Alignment will be disabled."
                )
        elif self.model_param_names is not None and "imline_mask_label" in self.model_param_names:
            logger.warning("Pre-training with Text-Image Alignment disabled (tiam_probability = 0)")

        return super().__post_init__()

    def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        raise NotImplementedError(
            "Custom Textract MLM data collator has not been implemented for NumPy"
        )

    def tf_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        raise NotImplementedError(
            "Custom Textract MLM data collator has not been implemented for TensorFlow"
        )

    def _is_tia_enabled(self) -> bool:
        """Safely check whether current configuration enables Text-Image Alignment (TIA) task"""
        return (
            self.model_param_names is not None
            and self.tiam_probability > 0
            and "imline_mask_label" in self.model_param_names
        )

    def _is_tim_enabled(self) -> bool:
        """Safely check whether current configuration enables Text-Image Matching (TIM) task"""
        return (
            self.model_param_names is not None
            and self.tim_probability > 0
            and "image_mask_label" in self.model_param_names
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

        # Masked Visual-Language Modelling (MVLM): Token masking as per the standard MLM superclass
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

        if "images" in batch:
            # Text Image Matching (TIM): Select which samples' images to randomly reassign:
            # (This fn is cheap to call if TIM is turned off)
            image_mask_labels, image_indices = self.torch_permute_images(tokenized["image"])

            if self._is_tia_enabled():
                # Text Image Alignment (TIA): For each image in the batch (including reassigned
                # ones), mask some text in the image.
                masked_images = []
                batch_tia_labels = []
                for ixex, is_image_permuted in enumerate(image_mask_labels):
                    image = tokenized["image"][image_indices[ixex]].clone()
                    tia_labels = self.torch_mask_lines_in_image(
                        image=image,
                        # Should word boxes/lids here be at ixex, or image_indices[ixex]? Using
                        # ixex produces masks aligned to current text. Using image_indices[ixex]
                        # produces masks aligned to the image. Here using `ixex` on the assumption
                        # it's easier for the model to shortcut learning whether masks misalign
                        # with word boxes, than whether masks misalign with image text lines.
                        word_line_ids=torch.LongTensor(batch["line-ids"][ixex]),
                        word_boxes=torch.LongTensor(batch["boxes"][ixex]),
                        is_permuted=is_image_permuted,
                        # TODO: Could maybe shortcut token_line_ids calc when is_image_permuted?
                        # For permuted images, all text tokens just get set as 'covered' anyway,
                        # but would still need to know which tokens are text.
                        token_line_ids=self._map_sample_line_ids(
                            word_line_ids=batch["line-ids"][ixex],
                            token_word_ids=tokenized.word_ids(ixex),
                        ),
                    )
                    # Ignore model TIA output for any tokens that are MVLM-masked:
                    tia_labels[tokenized["labels"][ixex] == -100] = -100
                    masked_images.append(image)
                    batch_tia_labels.append(tia_labels)

                # Update images and add image masking labels to the batch:
                tokenized["image"] = torch.stack(masked_images)
                tokenized["imline_mask_label"] = torch.stack(batch_tia_labels)
            else:
                # TIA is not enabled - just apply the TIM permutation:
                shuffled_images = tokenized["image"][image_indices]
                tokenized["image"] = shuffled_images

            if self._is_tim_enabled():
                tokenized["image_mask_label"] = image_mask_labels

        return tokenized

    def torch_permute_images(self, images: Iterable[Any]) -> Tuple[torch.LongTensor, Iterable[Any]]:
        """Randomly permute input image tensors between samples with tim_probability

        TODO: "Drop" images sometimes instead of swapping
        The LayoutLMv2 paper refers to permuting images most of the time but sometimes "dropping"
        them. That isn't implemented here yet.

        Parameters
        ----------
        images :
            Array of page images for the batch (after tokenization/processing), to be masked.

        Returns
        -------
        masked_indices :
            1 where the sample's image was masked (swapped randomly with another sample in the
            batch), and 0 where the returned image uses the original.
        selected_indices :
            The index of image from the batch that should be used for each sample.
        """
        batch_size = len(images)

        if batch_size < 2:
            # Can't permute on a batch that only has one sample:
            return (
                # masked_indices = [0] (not masked)
                torch.zeros((batch_size,), dtype=torch.long),
                # selected_indices = [0] (use the only image in batch)
                torch.zeros((batch_size,), dtype=torch.long),
            )
        elif self.tim_probability <= 0:
            return (
                # masked_indices = always 0 (not masked)
                torch.zeros((batch_size,), dtype=torch.long),
                # selected_indices = [0, 1, 2, ...]
                torch.arange(0, batch_size, dtype=torch.long),
            )

        # Select which images in the batch should be masked/swapped:
        probability_matrix = torch.full((batch_size,), self.tim_probability)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Select which image to sub in for masked positions, by rotating a random (nonzero) number
        # of samples around the batch:
        selected_indices = torch.arange(0, batch_size, dtype=torch.long)
        random_indices = (
            (selected_indices + torch.randint(low=1, high=batch_size, size=(batch_size,)))
            % batch_size
        ).long()
        selected_indices[masked_indices] = random_indices[masked_indices]

        return masked_indices.long(), selected_indices

    def torch_mask_lines_in_image(
        self,
        image: torch.Tensor,
        word_line_ids: torch.LongTensor,
        word_boxes: torch.LongTensor,
        is_permuted: bool,
        token_line_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        """Mask text lines from an image (in-place) and return token-level mask labels"""
        # Initial labels are that no tokens are masked if the image aligns to the sample, OR every
        # token is masked if the image got permuted:
        if is_permuted:
            token_imline_masked = torch.ones_like(token_line_ids, dtype=torch.long)
        else:
            token_imline_masked = torch.zeros_like(token_line_ids, dtype=torch.long)
        # ...But ignore model output for any non-text tokens (CLS, SEP, PAD, etc):
        token_imline_masked[token_line_ids < 0] = -100

        # Choose lines to mask from the de-duplicated list of line IDs:
        unique_line_ids = torch.unique(word_line_ids, sorted=False)
        masked_line_indices = torch.bernoulli(
            torch.full(unique_line_ids.size(), self.tiam_probability)
        ).bool()
        masked_line_ids = unique_line_ids[masked_line_indices]  # Array of line_ids to mask out

        # Apply each mask to the image & labels:
        for line_id in masked_line_ids:
            # Calculate the (4,) x0,y0,x1,y1 box of coordinates 0-1000 enclosing all words within
            # this line ID:
            line_word_boxes = word_boxes[word_line_ids == line_id, :]
            line_box = torch.cat(
                [
                    torch.min(line_word_boxes[:, :2], dim=0).values,  # Min of x0, Min of y0
                    torch.max(line_word_boxes[:, 2:], dim=0).values,  # Max of x1, Max of y1
                ]
            )
            # Map from normalized to image coordinates:
            n_channels, img_height, img_width = image.size()
            line_box = torch.round(
                line_box * torch.Tensor([img_width, img_height, img_width, img_height]) / 1000.0
            ).long()
            image[:, line_box[1] : line_box[3], line_box[0] : line_box[2]] = 0
            # Mark all tokens belonging to this line as masked (unless is_permuted, in which case
            # no need as every token is already labelled masked):
            if not is_permuted:
                token_imline_masked[token_line_ids == line_id] = 1

        # Return the labels (image was modified in-place):
        return token_imline_masked


def prepare_dataset(
    textract_path: str,
    tokenizer: PreTrainedTokenizerBase,
    manifest_file_path: str,
    images_path: Optional[str] = None,
    images_prefix: str = "",
    textract_prefix: str = "",
    output_line_ids: bool = True,
    max_seq_len: int = 512,
    num_workers: Optional[int] = None,
    batch_size: int = 16,
    cache_dir: Optional[str] = None,
    cache_file_prefix: Optional[str] = None,
):
    return split_long_dataset_samples(
        prepare_base_dataset(
            textract_path=textract_path,
            manifest_file_path=manifest_file_path,
            images_path=images_path,
            images_prefix=images_prefix,
            textract_prefix=textract_prefix,
            output_line_ids=output_line_ids,
            num_workers=num_workers,
            batch_size=batch_size,
            cache_dir=cache_dir,
            map_cache_file_name=(
                os.path.join(cache_dir, f"{cache_file_prefix}_1base.arrow")
                if (cache_dir and cache_file_prefix)
                else None
            ),
        ),
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        num_workers=num_workers,
        batch_size=batch_size,
        cache_file_name=(
            os.path.join(cache_dir, f"{cache_file_prefix}_2label.arrow")
            if (cache_dir and cache_file_prefix)
            else None
        ),
    )


def get_task(
    data_args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizerBase,
    processor: Optional[ProcessorMixin] = None,
    model_param_names: Optional[Iterable[str]] = None,
    n_workers: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> TaskData:
    """Load datasets and data collators for MLM model training"""
    logger.info("Getting MLM datasets")
    # We only need to track line IDs for each word when TIAM is enabled:
    if model_param_names is None:
        tiam_enabled = False
        logger.warning(
            "Skipping generation of text line IDs in dataset: Since model_param_names is not "
            "provided, we don't know if this field is required/supported by the model."
        )
    else:
        tiam_enabled = ("imline_mask_label" in model_param_names) and (
            data_args.tiam_probability > 0
        )
    train_dataset = prepare_dataset(
        data_args.textract,
        tokenizer=tokenizer,
        manifest_file_path=data_args.train,
        images_path=data_args.images,
        images_prefix=data_args.images_prefix,
        textract_prefix=data_args.textract_prefix,
        output_line_ids=tiam_enabled,
        max_seq_len=data_args.max_seq_length - 2,  # To allow for CLS+SEP in final
        num_workers=n_workers,
        batch_size=data_args.dataproc_batch_size,
        cache_dir=cache_dir,
        cache_file_prefix="mlmtrain",
    )
    logger.info("Train dataset ready: %s", train_dataset)

    if data_args.validation:
        eval_dataset = prepare_dataset(
            data_args.textract,
            tokenizer=tokenizer,
            manifest_file_path=data_args.validation,
            images_path=data_args.images,
            images_prefix=data_args.images_prefix,
            textract_prefix=data_args.textract_prefix,
            output_line_ids=tiam_enabled,
            max_seq_len=data_args.max_seq_length - 2,  # To allow for CLS+SEP in final
            num_workers=n_workers,
            batch_size=data_args.dataproc_batch_size,
            cache_dir=cache_dir,
            cache_file_prefix="mlmval",
        )
        logger.info("Validation dataset ready: %s", eval_dataset)
    else:
        eval_dataset = None

    return TaskData(
        train_dataset=train_dataset,
        data_collator=TextractLayoutLMDataCollatorForLanguageModelling(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=data_args.pad_to_multiple_of,
            processor=processor,
            model_param_names=model_param_names,
            tiam_probability=data_args.tiam_probability,
            tim_probability=data_args.tim_probability,
        ),
        eval_dataset=eval_dataset,
        metric_computer=None,
    )
