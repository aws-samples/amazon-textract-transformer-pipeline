# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Entity recognition (token classification) dataset classes for Textract+LayoutLM

This implementation trains a token classification model using a SageMaker Ground Truth Object
Detection manifest: By mapping the annotated boxes on page images to the positions of WORD blocks
extracted by Textract, to classify each WORD to an entity type (or 'other' if no boxes tag it).
"""
# Python Built-Ins:
from dataclasses import dataclass
from numbers import Real
import os
from typing import Any, Callable, Dict, List, Optional, Union

# External Dependencies:
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers.data.data_collator import DataCollatorForTokenClassification
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# Local Dependencies:
from ..config import DataTrainingArguments
from ..logging_utils import getLogger
from .base import (
    LayoutLMDataCollatorMixin,
    prepare_base_dataset,
    split_long_dataset_samples,
    TaskData,
)
from .smgt import BoundingBoxAnnotationResult


logger = getLogger("data.ner")


def word_label_matrix_from_norm_bounding_boxes(
    boxes: np.array,  # TODO: PyTorch Tensor support?
    smgt_boxes_ann: BoundingBoxAnnotationResult,
    n_classes: int,
) -> np.ndarray:
    """Calculate (multi-label) word classes by overlap of Textract (TRP) items with SMGT BBoxes

    Parameters
    ----------
    textract_blocks :
        List-like of TRP objects with a 'geometry' property (e.g. Word, Line, Cell, Page, etc)
    smgt_boxes_ann :
        Parsed result from a SageMaker Ground Truth Bounding Box annotation job
    n_classes :
        Number of classes in the annotation dataset

    Returns
    -------
    result : np.array
        (n_textract_blocks, n_classes) matrix of 0|1 class labels, defined as 1 where half or more
        of the block's area intersects with a bounding box annotation of that class. Note that
        multi-classification is supported so rows may sum to more than 1.
    """
    n_words = len(boxes)  # (n_words, 4) 0-1000 x0, y0, x1, y1
    ann_boxes = smgt_boxes_ann.normalized_boxes(return_tensors="np")
    if len(ann_boxes) == 0:
        # Easier to just catch this case than deal with it later:
        return np.concatenate(
            [np.zeros((n_words, n_classes - 1)), np.ones((n_words, 1))],
            axis=1,
        )
    ann_class_ids = np.array([box.class_id for box in smgt_boxes_ann.boxes])
    n_anns = len(ann_boxes)  # (n_words, 4) 0-1000 x0, y0, x1, y1

    word_widths = boxes[:, 2] - boxes[:, 0]
    word_heights = boxes[:, 3] - boxes[:, 1]
    word_areas = word_widths * word_heights

    # We want to produce a matrix (n_words, n_anns) describing overlaps
    # Note the slicing e.g. boxes[:, 2:3] vs boxes[:, 2] keeps the result a vector instead of 1D
    # array (as 1D array would mess up the tiling)
    isects_right = np.minimum(
        np.tile(boxes[:, 2:3], (1, n_anns)),
        np.tile(ann_boxes[:, 2:3].transpose(), (n_words, 1)),
    )
    isects_left = np.maximum(
        np.tile(boxes[:, 0:1], (1, n_anns)),
        np.tile(ann_boxes[:, 0:1].transpose(), (n_words, 1)),
    )
    isects_width = np.maximum(0, isects_right - isects_left)
    isects_bottom = np.minimum(
        np.tile(boxes[:, 3:4], (1, n_anns)),
        np.tile(ann_boxes[:, 3:4].transpose(), (n_words, 1)),
    )
    isects_top = np.maximum(
        np.tile(boxes[:, 1:2], (1, n_anns)),
        np.tile(ann_boxes[:, 1:2].transpose(), (n_words, 1)),
    )
    isects_height = np.maximum(0, isects_bottom - isects_top)

    matches = np.where(
        # (Need to convert word_areas from 1D array to column vector)
        isects_width * isects_height >= (word_areas / 2)[:, np.newaxis],
        1.0,
        0.0,
    )

    # But `matches` is not the final result: We want a matrix by class IDs, not every bounding box
    result = np.zeros((n_words, n_classes))
    # Not aware of any way to do this without looping yet:
    for class_id in range(n_classes):
        class_matches = np.any(matches[:, ann_class_ids == class_id], axis=1)
        result[:, class_id] = class_matches
    # Implicitly any word with no matches is "other", class n-1:
    result[:, n_classes - 1] = np.where(
        np.any(result, axis=1),
        result[:, n_classes - 1],
        1.0,
    )

    return result


def word_single_labels_from_norm_bounding_boxes(
    boxes: np.array,
    smgt_boxes_ann: BoundingBoxAnnotationResult,
    n_classes: int,
):
    """Assign a class label to each Textract (TRP) block based on SMGT Bounding Box overlaps

    Any words matching multiple classes' annotations are silently tagged to the lowest matched
    class index. Words with no annotation are tagged to the 'other' class (n_classes - 1).

    Parameters
    ----------
    boxes :
        Array of normalized LayoutLM-like word bounding boxes
    smgt_boxes_ann :
        Parsed result from a SageMaker Ground Truth Bounding Box annotation job
    n_classes :
        Number of classes in the annotation dataset INCLUDING the 'other' class (implicitly =
        n_classes - 1)

    Returns
    -------
    result : np.array
        (n_textract_blocks,) array of class label integers by word, from 0 to n_classes - 1
        inclusive.
    """
    word_labels = word_label_matrix_from_norm_bounding_boxes(boxes, smgt_boxes_ann, n_classes)
    return np.where(
        np.sum(word_labels, axis=1) == 0,
        n_classes - 1,
        np.argmax(word_labels, axis=1),
    )


def map_smgt_boxes_to_word_labels(
    batch: Dict[str, List],  # TODO: Support List[Any]? Union[Dict[List], List[Any]],
    # TODO: Any way to link to original manifest lines for error diagnostics?
    annotation_attr: str,
    n_classes: int,
):
    """datasets.map function to tag "word_labels" from word "boxes" and SMGT bbox annotation data"""
    # TODO: Check if manifest-line diagnostic feed-through is actually working? Seems broken
    manifest_lines = batch.get("manifest-line")
    if annotation_attr not in batch:
        raise ValueError(
            "Bounding box label attribute '{}' missing from batch{}".format(
                annotation_attr, " (Manifest lines {manifest_lines})." if manifest_lines else "."
            )
        )
    # TODO: More useful error messages if one fails?
    annotations = [BoundingBoxAnnotationResult(ann) for ann in batch[annotation_attr]]
    word_labels = [
        word_single_labels_from_norm_bounding_boxes(
            # Although layoutlm_boxes_from_trp_blocks originally creates these as numpy, they seem
            # to get converted back to listed lists in the dataset. Therefore re-numpying:
            np.array(boxes),
            annotations[i],
            n_classes,
        )
        for i, boxes in enumerate(batch["boxes"])
    ]
    result = {k: v for k, v in batch.items()}
    result["word_labels"] = word_labels
    return result


@dataclass
class TextractLayoutLMDataCollatorForWordClassification(
    LayoutLMDataCollatorMixin,
    DataCollatorForTokenClassification,
):
    """Collator to process (batches of) Examples into batched model inputs

    For our case, tokenization can happen at the batch level which allows us to pad to the longest
    sample in batch rather than the overall model max_seq_len. Word splitting is already done by
    Textract, and some custom logic is required to feed through the bounding box inputs from
    Textract (at word level) to the model inputs (at token level).
    """

    def __post_init__(self):
        self._init_for_layoutlm()
        # super().__post_init__ not present in this parent class

    def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        raise NotImplementedError(
            "Custom Textract NER data collator has not been implemented for NumPy"
        )

    def tf_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        raise NotImplementedError(
            "Custom Textract NER data collator has not been implemented for TensorFlow"
        )

    def torch_call(
        self,
        batch: Union[Dict[str, List], List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        if isinstance(batch, list):
            batch = {k: [ex[k] for ex in batch] for k in batch[0]}

        # First tokenize/preprocess the batch, then we'll perform any fixes needed for NER:
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

        # LayoutLMV1Tokenizer doesn't map through "word_labels", so we need to do it ourselves:
        # (`labels` is same forward() param for all ForTokenClassification model versions)
        if "word_labels" in batch and "labels" not in tokenized:
            label_tensors_by_example = []
            n_examples = len(batch["text"])
            n_example_words = [len(words) for words in batch["text"]]

            for ixex in range(n_examples):
                word_ids = tokenized.word_ids(ixex)
                n_words = n_example_words[ixex]

                # We map labels by augmenting the list of word labels to include values for the
                # special tokens, editing the word_ids mapping from tokens->words to match special
                # tokens to their special values (instead of None), and then applying this set of
                # indexes to produce the token-wise labels.
                augmented_example_labels = torch.LongTensor(
                    np.concatenate((batch["word_labels"][ixex], [self.label_pad_token_id]))
                )
                # Torch tensors don't support None->NaN, but numpy float ndarrays do:
                token_label_ids_np = np.array(word_ids, dtype=float)
                token_label_ids_np[np.isnan(token_label_ids_np)] = n_words
                label_tensors_by_example.append(
                    torch.index_select(
                        augmented_example_labels, 0, torch.LongTensor(token_label_ids_np)
                    )
                )

            tokenized["labels"] = torch.stack(label_tensors_by_example)

        # LayoutLMV1Tokenizer also doesn't map through "boxes", but this is common across tasks so
        # it's implemented in the parent mixin:
        self._map_word_boxes(tokenized, batch["boxes"])

        # We don't add extra tracability attributes here (e.g. page_num, textract_block_ids) since
        # the output still has a word_ids() function enabling inference users to do that for
        # themselves at inference time if needed.
        return tokenized


def prepare_dataset(
    textract_path: str,
    tokenizer: PreTrainedTokenizerBase,
    manifest_file_path: str,
    annotation_attr: str,
    n_classes: int,
    images_path: Optional[str] = None,
    images_prefix: str = "",
    textract_prefix: str = "",
    max_seq_len: int = 512,
    num_workers: Optional[int] = None,
    batch_size: int = 16,
    cache_dir: Optional[str] = None,
    cache_file_prefix: Optional[str] = None,
):
    ds = prepare_base_dataset(
        textract_path=textract_path,
        manifest_file_path=manifest_file_path,
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
        map_smgt_boxes_to_word_labels,
        batched=True,
        batch_size=batch_size,
        fn_kwargs={
            "annotation_attr": annotation_attr,
            "n_classes": n_classes,
        },
        num_proc=num_workers,
        desc="Reconciling bounding boxes to Textract words",
        cache_file_name=(
            os.path.join(cache_dir, f"{cache_file_prefix}_2label.arrow")
            if (cache_dir and cache_file_prefix)
            else None
        ),
    )

    return split_long_dataset_samples(
        ds,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        num_workers=num_workers,
        batch_size=batch_size,
        cache_file_name=(
            os.path.join(cache_dir, f"{cache_file_prefix}_3split.arrow")
            if (cache_dir and cache_file_prefix)
            else None
        ),
    )


def get_metric_computer(
    num_labels: int,
    pad_token_label_id: int = CrossEntropyLoss().ignore_index,
) -> Callable[[Any], Dict[str, Real]]:
    """Generate token classification compute_metrics callable for a HF Trainer

    Note that we don't use seqeval (like some standard TokenClassification examples) because our
    tokens are not guaranteed to be in reading order by the nature of LayoutLM/OCR: So conventional
    Inside-Outside-Beginning (IOB/ES) notation doesn't really make sense.
    """
    other_class_label = num_labels - 1

    def compute_metrics(p: Any) -> Dict[str, Real]:
        probs_raw, labels_raw = p
        predicted_class_ids_raw = np.argmax(probs_raw, axis=2)

        # Override padding token predictions to ignore value:
        non_pad_labels = labels_raw != pad_token_label_id
        predicted_class_ids_raw = np.where(
            non_pad_labels,
            predicted_class_ids_raw,
            pad_token_label_id,
        )

        # Update predictions by label:
        unique_labels, unique_counts = np.unique(predicted_class_ids_raw, return_counts=True)

        # Accuracy ignoring PAD, CLS and SEP tokens:
        n_tokens_by_example = non_pad_labels.sum(axis=1)
        n_tokens_total = n_tokens_by_example.sum()
        n_correct_by_example = np.logical_and(
            labels_raw == predicted_class_ids_raw, non_pad_labels
        ).sum(axis=1)
        acc_by_example = np.true_divide(n_correct_by_example, n_tokens_by_example)

        # Accuracy ignoring PAD, CLS, SEP tokens *and* tokens where both pred and actual classes
        # are 'other':
        focus_labels = np.logical_and(
            non_pad_labels,
            np.logical_or(
                labels_raw != other_class_label,
                predicted_class_ids_raw != other_class_label,
            ),
        )
        n_focus_tokens_by_example = focus_labels.sum(axis=1)
        n_focus_correct_by_example = np.logical_and(
            labels_raw == predicted_class_ids_raw,
            focus_labels,
        ).sum(axis=1)
        focus_acc_by_example = np.true_divide(
            n_focus_correct_by_example[n_focus_tokens_by_example != 0],
            n_focus_tokens_by_example[n_focus_tokens_by_example != 0],
        )
        logger.info(
            "Evaluation class prediction ratios: {}".format(
                {
                    unique_labels[ix]: unique_counts[ix] / n_tokens_total
                    for ix in range(len(unique_counts))
                    if unique_labels[ix] != pad_token_label_id
                }
            )
        )
        n_examples = probs_raw.shape[0]
        acc = acc_by_example.sum() / n_examples

        n_focus_examples = n_focus_tokens_by_example[n_focus_tokens_by_example != 0].shape[0]
        focus_acc = focus_acc_by_example.sum() / n_focus_examples

        return {
            "n_examples": n_examples,
            "acc": acc,
            "n_focus_examples": n_focus_examples,
            "focus_acc": focus_acc,
            # By nature of the metric, focus_acc can sometimes take a few epochs to move away from
            # 0.0. Since acc and focus_acc are both 0-1, we can define this metric to show early
            # improvement (thus prevent early stopping) but still target focus_acc later:
            "focus_else_acc_minus_one": focus_acc if focus_acc > 0 else acc - 1,
        }

    return compute_metrics


def get_task(
    data_args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizerBase,
    processor: Optional[ProcessorMixin] = None,
    n_workers: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> TaskData:
    """Load datasets and data collators for NER model training"""
    logger.info("Getting NER datasets")

    train_dataset = prepare_dataset(
        data_args.textract,
        tokenizer=tokenizer,
        manifest_file_path=data_args.train,
        annotation_attr=data_args.annotation_attr,
        n_classes=data_args.num_labels,
        images_path=data_args.images,
        images_prefix=data_args.images_prefix,
        textract_prefix=data_args.textract_prefix,
        max_seq_len=data_args.max_seq_length - 2,  # To allow for CLS+SEP in final
        num_workers=n_workers,
        batch_size=data_args.dataproc_batch_size,
        cache_dir=cache_dir,
        cache_file_prefix="nertrain",
    )
    logger.info("Train dataset ready: %s", train_dataset)

    if data_args.validation:
        eval_dataset = prepare_dataset(
            data_args.textract,
            tokenizer=tokenizer,
            manifest_file_path=data_args.validation,
            annotation_attr=data_args.annotation_attr,
            n_classes=data_args.num_labels,
            images_path=data_args.images,
            images_prefix=data_args.images_prefix,
            textract_prefix=data_args.textract_prefix,
            max_seq_len=data_args.max_seq_length - 2,  # To allow for CLS+SEP in final
            num_workers=n_workers,
            batch_size=data_args.dataproc_batch_size,
            cache_dir=cache_dir,
            cache_file_prefix="nerval",
        )
        logger.info("Validation dataset ready: %s", eval_dataset)
    else:
        eval_dataset = None

    return TaskData(
        train_dataset=train_dataset,
        data_collator=TextractLayoutLMDataCollatorForWordClassification(
            tokenizer=tokenizer,
            pad_to_multiple_of=data_args.pad_to_multiple_of,
            processor=processor,
        ),
        eval_dataset=eval_dataset,
        metric_computer=get_metric_computer(data_args.num_labels),
    )
