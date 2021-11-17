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
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Optional, Tuple, Type, Union

# External Dependencies:
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers.data.data_collator import DataCollatorForTokenClassification
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import trp

# Local Dependencies:
from ..config import DataTrainingArguments
from ..logging_utils import getLogger
from .base import ExampleSplitterBase, NaiveExampleSplitter, TaskData, TextractLayoutLMDatasetBase
from .geometry import BoundingBoxAnnotationResult, layoutlm_boxes_from_trp_blocks


logger = getLogger("data.ner")


@dataclass
class TextractLayoutLMExampleForWordClassification:
    """Data class yielded as examples by an NER dataset for training or inference"""

    word_boxes_normalized: np.ndarray  # Nx4 array already normalized to LayoutLM 0-1000 format
    word_texts: List[str]  # List of length N, text per Textract WORD block
    word_labels: Optional[np.ndarray] = None  # 1D integer array of length N, class ID per word


@dataclass
class TextractLayoutLMDataCollatorForWordClassification(DataCollatorForTokenClassification):
    """Collator to process (batches of) Examples into batched model inputs

    For our case, tokenization can happen at the batch level which allows us to pad to the longest
    sample in batch rather than the overall model max_seq_len. Word splitting is already done by
    Textract, and some custom logic is required to feed through the bounding box inputs from
    Textract (at word level) to the model inputs (at token level).
    """

    bos_token_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    pad_token_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    sep_token_box: Tuple[int, int, int, int] = (1000, 1000, 1000, 1000)

    def __post_init__(self):
        self.special_token_boxes = torch.LongTensor(
            [
                self.bos_token_box,
                self.pad_token_box,
                self.sep_token_box,
            ]
        )
        # super().__post_init__ not present in this parent class

    def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        raise NotImplementedError(
            "Custom Textract MLM data collator has not been implemented for NumPy"
        )

    def tf_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        raise NotImplementedError(
            "Custom Textract MLM data collator has not been implemented for TensorFlow"
        )

    def torch_call(
        self,
        examples: List[TextractLayoutLMExampleForWordClassification],
    ) -> Dict[str, Any]:
        # Tokenize, pad and etc the words:
        batch = self.tokenizer(
            [example.word_texts for example in examples],
            is_split_into_words=True,
            return_attention_mask=True,
            padding=bool(self.pad_to_multiple_of),
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        # Map through the bounding boxes and word labels to the generated tokens:
        # We do this by augmenting the list of word bboxes/labels to include values for the special
        # tokens, editing the word_ids mapping from tokens->words to match special tokens to their
        # special values (instead of None), and then applying this set of indexes to produce the
        # token-wise boxes/labels including special tokens.
        bbox_tensors_by_example = []
        label_tensors_by_example = []
        for ixex in range(len(examples)):
            word_ids = batch.word_ids(ixex)
            n_real_words = len(examples[ixex].word_texts)

            # Map labels:
            # (Labels have just one dummy label for all padding/special tokens)
            if examples[ixex].word_labels is not None:
                augmented_example_labels = torch.LongTensor(
                    np.concatenate((examples[ixex].word_labels, [self.label_pad_token_id]))
                )
                # Torch tensors don't support None->NaN, but numpy float ndarrays do:
                token_label_ids_np = np.array(word_ids, dtype=float)
                token_label_ids_np[np.isnan(token_label_ids_np)] = n_real_words
                label_tensors_by_example.append(
                    torch.index_select(
                        augmented_example_labels, 0, torch.LongTensor(token_label_ids_np)
                    )
                )
            elif len(label_tensors_by_example):
                raise ValueError(
                    "word_labels must be present in all or none of examples in batch, but example "
                    "%s has no labels" % ixex
                )

            # Map boxes:
            # (Boxes have different special values for different special tokens)
            augmented_example_word_boxes = torch.cat(
                (
                    torch.LongTensor(examples[ixex].word_boxes_normalized),
                    self.special_token_boxes,
                ),
                dim=0,
            )
            # Torch tensors don't support None->NaN, but numpy float ndarrays do:
            box_ids_np = np.array(word_ids, dtype=float)
            box_ids_np = np.where(
                batch.input_ids[ixex, :] == self.tokenizer.bos_token_id,
                n_real_words,  # bos_token_box, per special_token_boxes
                box_ids_np,
            )
            box_ids_np = np.where(
                batch.input_ids[ixex, :] == self.tokenizer.cls_token_id,
                n_real_words,  # cls_token_box, per special_token_boxes
                box_ids_np,
            )
            box_ids_np = np.where(
                batch.input_ids[ixex, :] == self.tokenizer.pad_token_id,
                n_real_words + 1,  # pad_token_box, per special_token_boxes
                box_ids_np,
            )
            box_ids_np = np.where(
                batch.input_ids[ixex, :] == self.tokenizer.sep_token_id,
                n_real_words + 2,  # sep_token_box, per special_token_boxes
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
        if len(label_tensors_by_example):
            batch["labels"] = torch.stack(label_tensors_by_example)

        # We don't add extra tracability attributes here (e.g. page_num, textract_block_ids) since
        # the output still has a word_ids() function enabling inference users to do that for
        # themselves.
        return batch


def word_label_matrix_from_bounding_boxes(
    textract_blocks: Iterable[
        Union[
            trp.Word,
            trp.Line,
            trp.SelectionElement,
            trp.FieldKey,
            trp.FieldValue,
            trp.Cell,
            trp.Table,
            trp.Page,
        ]
    ],
    smgt_boxes_ann: BoundingBoxAnnotationResult,
    n_classes: int,
):
    """Calculate overlap of Textract (TRP) items with SMGT BBoxes to multi-label word classes

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
    word_boxes = [word.geometry.boundingBox for word in textract_blocks]
    result = np.zeros((len(word_boxes), n_classes))
    for ixword, wordbox in enumerate(word_boxes):
        word_area = wordbox.height * wordbox.width
        for annbox in smgt_boxes_ann.boxes:
            isect_area = (
                # Intersection width:
                max(
                    0,
                    # Intersection right:
                    min(wordbox.left + wordbox.width, annbox.rel_right)
                    # Minus intersection left:
                    - max(wordbox.left, annbox.rel_left),
                )
                # Intersection height:
                * max(
                    0,
                    # Intersection bottom:
                    min(wordbox.top + wordbox.height, annbox.rel_bottom)
                    # Minus intersection top:
                    - max(wordbox.top, annbox.rel_top),
                )
            )
            if isect_area >= (word_area / 2):
                result[ixword, annbox.class_id] = 1.0
    return result


def word_single_labels_from_bounding_boxes(
    textract_blocks: Iterable[
        Union[
            trp.Word,
            trp.Line,
            trp.SelectionElement,
            trp.FieldKey,
            trp.FieldValue,
            trp.Cell,
            trp.Table,
            trp.Page,
        ]
    ],
    smgt_boxes_ann: BoundingBoxAnnotationResult,
    n_classes: int,
):
    """Assign a class label to each Textract (TRP) block based on SMGT Bounding Box overlaps

    Any words matching multiple classes' annotations are silently tagged to the lowest matched
    class index. Words with no annotation are tagged to the 'other' class (n_classes - 1).

    Parameters
    ----------
    textract_blocks :
        List-like of TRP objects with a 'geometry' property (e.g. Word, Line, Cell, Page, etc)
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
    word_labels = word_label_matrix_from_bounding_boxes(textract_blocks, smgt_boxes_ann, n_classes)
    return np.where(
        np.sum(word_labels, axis=1) == 0,
        n_classes - 1,
        np.argmax(word_labels, axis=1),
    )


class TextractLayoutLMDatasetForTokenClassification(TextractLayoutLMDatasetBase):
    """PyTorch/Hugging Face dataset for token classification/NER with LayoutLM & Amazon Textract

    This implementation parses the dataset up-front to calculate an overall length (in number of
    examples, after splitting any long pages) and enable random access - rather than having to
    worry about implementing order randomization in a streaming API.

    We assume the dataset (as represented in example_index) fits into available memory for
    simplicity and performance. If that's not the case, you could consider other options - for
    example serializing example_index out to disk and reloading entries on-demand in __getitem__.

    This dataset operates on JSONLines manifest files with records in the following form:

    record["textract-ref"] : str
        S3 URI to a response JSON from Amazon Textract (not 'source-ref', because you needed to use
        that for the actual page image per SMGT).
    record["page-num"] : int (optional)
        *1-based* page number of the Textract document that this record refers to
    record[annnotation_attr] : Dict
        An annotation record in similar format to what would be generated by a SageMaker Ground
        Truth Object Detection job, as required to parse a BoundingBoxAnnotationResult
    """

    splitter: ClassVar[Type[ExampleSplitterBase]] = NaiveExampleSplitter

    def __init__(
        self,
        textract_path: str,
        tokenizer: PreTrainedTokenizerBase,
        manifest_file_path: str,
        num_labels: int,
        annotation_attr: str,
        textract_prefix: str = "",
        max_seq_len: int = 512,
    ):
        """Initialize a TextractLayoutLMDatasetForTokenClassification

        Arguments
        ---------
        num_labels : int
            Number of entity classes to classify tokens between, *including* the implicit "other"
            class
        annotation_attr : str
            Attribute on the input manifest file where SageMaker Ground Truth-compatible bounding
            box annotations are stored.

        Additional arguments as per TextractLayoutLMDatasetBase
        """
        super().__init__(
            textract_path,
            tokenizer,
            manifest_file_path=manifest_file_path,
            textract_prefix=textract_prefix,
            max_seq_len=max_seq_len,
        )

        self.num_labels = num_labels
        self.example_index: List[TextractLayoutLMExampleForWordClassification] = []

        # Load the raw manifest data:
        for ipagespec, pagespec in enumerate(self.dataset_inputs(), start=1):
            textract_file_path = pagespec["textract-ref"]
            textract_doc = self.parse_textract_file(textract_file_path)

            if "page-num" in pagespec:
                if len(textract_doc.pages) < pagespec["page-num"]:
                    raise ValueError(
                        "".join(
                            (
                                "(Manifest line {}) page-num {} out of range of Textract result ",
                                "pages ({})",
                            )
                        ).format(
                            ipagespec,
                            pagespec["page-num"],
                            len(textract_doc.pages),
                        )
                    )
            elif len(textract_doc.pages) != 1:
                raise ValueError(
                    "".join(
                        (
                            "(Manifest line {}) got {} pages in Textract file (exactly 1 ",
                            "required when page-num not specified)",
                        )
                    ).format(
                        ipagespec,
                        len(textract_doc.pages),
                    )
                )

            if annotation_attr not in pagespec:
                raise ValueError(
                    "(Manifest line {}) annotation_attr '{}' is missing".format(
                        ipagespec,
                        annotation_attr,
                    )
                )
            annotation = BoundingBoxAnnotationResult(pagespec[annotation_attr])
            page = textract_doc.pages[pagespec["page-num"] - 1]  # (page-num is 1-based)
            words = [word for line in page.lines for word in line.words]
            word_labels = word_single_labels_from_bounding_boxes(words, annotation, num_labels)
            word_texts = [word.text for word in words]
            page_splits = self.splitter.split(
                word_texts,
                tokenizer,
                self.max_content_seq_len,
            )
            for start_word, end_word in page_splits:
                self.example_index.append(
                    TextractLayoutLMExampleForWordClassification(
                        word_boxes_normalized=layoutlm_boxes_from_trp_blocks(
                            words[start_word:end_word],
                        ),
                        word_texts=word_texts[start_word:end_word],
                        word_labels=word_labels[start_word:end_word],
                    )
                )
            logger.debug("(Manifest line %s) generated %s examples", ipagespec, page_splits)

    def __getitem__(self, idx: int) -> TextractLayoutLMExampleForWordClassification:
        return self.example_index[idx]

    def __len__(self) -> int:
        return len(self.example_index)


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
        focus_acc = focus_acc_by_example.sum() / n_examples
        return {
            "n_examples": n_examples,
            "acc": acc,
            "focus_acc": focus_acc,
            # By nature of the metric, focus_acc can sometimes take a few epochs to move away from
            # 0.0. Since acc and focus_acc are both 0-1, we can define this metric to show early
            # improvement (thus prevent early stopping) but still target focus_acc later:
            "focus_else_acc_minus_one": focus_acc if focus_acc > 0 else acc - 1,
        }

    return compute_metrics


def get_task(
    data_args: DataTrainingArguments,
    tokenizer,
) -> TaskData:
    """Load datasets and data collators for NER model training"""
    train_dataset = TextractLayoutLMDatasetForTokenClassification(
        data_args.textract,
        tokenizer,
        data_args.train,
        data_args.num_labels,
        data_args.annotation_attr,
        max_seq_len=data_args.max_seq_length,
        textract_prefix=data_args.textract_prefix,
    )
    eval_dataset = (
        TextractLayoutLMDatasetForTokenClassification(
            data_args.textract,
            tokenizer,
            data_args.validation,
            data_args.num_labels,
            data_args.annotation_attr,
            max_seq_len=data_args.max_seq_length,
            textract_prefix=data_args.textract_prefix,
        )
        if data_args.validation
        else None
    )

    return TaskData(
        train_dataset=train_dataset,
        data_collator=TextractLayoutLMDataCollatorForWordClassification(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
        ),
        eval_dataset=eval_dataset,
        metric_computer=get_metric_computer(data_args.num_labels),
    )
