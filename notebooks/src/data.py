# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""SageMaker data loading utilities for Amazon Textract LayoutLM"""

# Python Built-Ins:
from dataclasses import dataclass
import json
import logging
import os
import re
from typing import Iterable, List, Union

# External Dependencies:
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset

# Local Dependencies:
import trp

logger = logging.getLogger("data")


def get_lines_in_reading_order(trp_lines: Iterable[trp.Line]) -> List[trp.Line]:
    """Sort a set of Textract Result Parser 'Line' objects in reading order.

    Re-implementing the trp.Page.getLinesInReadingOrder() logic which unfortunately returns text
    rather than the full trp.Line objects.
    """
    columns = []
    lines = []

    for item in trp_lines:
        column_found = False
        bbox_left = item.geometry.boundingBox.left
        bbox_right = item.geometry.boundingBox.left + item.geometry.boundingBox.width
        bbox_centre = (bbox_left + bbox_right) / 2
        for index, column in enumerate(columns):
            column_centre = (column["left"] + column["right"]) / 2
            if (bbox_centre > column["left"] and bbox_centre < column["right"]) or (
                column_centre > bbox_left and column_centre < bbox_right
            ):
                # BBox appears inside the column
                lines.append([index, item])
                column_found = True
                break
        if not column_found:
            columns.append(
                {
                    "left": item.geometry.boundingBox.left,
                    "right": item.geometry.boundingBox.left + item.geometry.boundingBox.width,
                }
            )
            lines.append([len(columns) - 1, item])
    lines.sort(key=lambda x: x[0])
    return list(map(lambda x: x[1], lines))


def get_layoutlm_bounding_boxes(
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
    return_tensors: Union["np", "pt"] = "np",
):
    """List of TRP 'blocks' to array of 0-1000 normalized x0,y0,x1,y1 for LayoutLM

    Per https://docs.aws.amazon.com/textract/latest/dg/API_BoundingBox.html, Textract bounding box
    coords are 0-1 relative to page size already: So we just need to multiply by 1000. Note this
    means there's no information encoded about the overall aspect ratio of the page.

    Parameters
    ----------
    textract_blocks :
        Iterable of any Textract TRP objects including a 'geometry' property e.g. Word, Line, Cell,
        etc.
    return_tensors : str (optional)
        Either "np" (default) to return a numpy array or "pt" to return a torch.LongTensor.

    Returns
    -------
    boxes :
        Array or tensor shape (n_examples, 4) of bounding boxes: left, top, right, bottom scaled
        0-1000.
    """
    raw_zero_to_one_list = [
        [
            block.geometry.boundingBox.left,
            block.geometry.boundingBox.top,
            block.geometry.boundingBox.left + block.geometry.boundingBox.width,
            block.geometry.boundingBox.top + block.geometry.boundingBox.height,
        ]
        for block in textract_blocks
    ]
    if return_tensors == "np":
        return np.array(raw_zero_to_one_list) * 1000
    elif reurn_tensors == "pt":
        # Easiest is to still go via numpy for the x1000 above:
        return (torch.FloatTensor(raw_zero_to_one_list) * 1000).long()


def group_bbox_annotations(smgt_annotation):
    """Bonus (not yet used/tested) fn for grouping overlapping SMGT bbox annotations of same class

    Parameters
    ----------
    smgt_annotation : SMGT.ObjectDetectionOutput
        Result field contents for an annotated object (per SageMaker Ground Truth Object Detection
        workflow)

    Returns
    -------
    groups : List[Dict{"class_id": int, "boxes": List[Dict]}]
        Grouping the 'annotations' from SMGT and moving the class_id key up to group level,
        preserving all other keys at the box level.
    """
    groups = []
    for annbox in smgt_annotation["annotations"]:
        class_id = annbox["class_id"]
        ann_matched_group = None
        new_groups = []  # Our new box might merge multiple groups!
        for group in filter(lambda g: g["class_id"] == class_id, groups):
            group_matches = False
            if group["class_id"] == class_id:
                for prevbox in group["boxes"]:
                    isect_top = max(annbox["top"], prevbox["top"])
                    isect_bottom = min(
                        annbox["top"] + annbox["height"],
                        prevbox["top"] + prevbox["height"],
                    )
                    if isect_bottom <= isect_top:
                        # We can already rule out intersection without computing X.
                        # (Calculate Y first assuming usually horizontal lines of text!)
                        continue
                    isect_left = max(annbox["left"], prevbox["left"])
                    isect_right = min(
                        annbox["left"] + annbox["width"],
                        prevbox["left"] + prevbox["width"],
                    )
                    if isect_right > isect_left:
                        # `annbox` overlaps with `group` -> Is part of the group. Stop testing
                        group_matches = True
                        break
            if group_matches:
                if ann_matched_group is None:
                    # The annotation matches this group and has not previously matched any other
                    # groups. Propagate this group to the next round.
                    new_groups.append(group)
                    ann_matched_group = group
                else:
                    # The annotation matches this group but has already matched another one. Merge:
                    ann_matched_group["boxes"] = ann_matched_group["boxes"] + group["boxes"]
            else:
                # Group doesn't match this box - just pass it through to the next box's search
                new_groups.append(group)
        if ann_matched_group is None:
            # This annotation didn't match any groups: Add a group for it:
            new_groups.append(
                {
                    "class_id": class_id,
                    "boxes": [{k: annbox[k] for k in annbox if k != "class_id"}],
                }
            )
        else:
            # Add this annotation as bounding box for the matched group:
            ann_matched_group["boxes"].append({k: annbox[k] for k in annbox if k != "class_id"})
        groups = new_groups
    # Looped through all annotations: groups are final & ready to return
    return groups


class AnnotationBoundingBox:
    """Data class to parse a bounding box annotated by SageMaker Ground Truth Object Detection

    Calculates all box TLHWBR metrics both absolute and relative on init, for efficient and easy
    processing later.
    """

    def __init__(self, manifest_box: dict, image_height: int, image_width: int):
        self._class_id = manifest_box["class_id"]
        self._abs_top = manifest_box["top"]
        self._abs_left = manifest_box["left"]
        self._abs_height = manifest_box["height"]
        self._abs_width = manifest_box["width"]
        self._abs_bottom = self.abs_top + self.abs_height
        self._abs_right = self.abs_left + self.abs_width
        self._rel_top = self._abs_top / image_height
        self._rel_left = self._abs_left / image_width
        self._rel_height = self._abs_height / image_height
        self._rel_width = self._abs_width / image_width
        self._rel_bottom = self._abs_bottom / image_height
        self._rel_right = self._abs_right / image_width

    @property
    def class_id(self):
        return self._class_id

    @property
    def abs_top(self):
        return self._abs_top

    @property
    def abs_left(self):
        return self._abs_left

    @property
    def abs_height(self):
        return self._abs_height

    @property
    def abs_width(self):
        return self._abs_width

    @property
    def abs_bottom(self):
        return self._abs_bottom

    @property
    def abs_right(self):
        return self._abs_right

    @property
    def rel_top(self):
        return self._rel_top

    @property
    def rel_left(self):
        return self._rel_left

    @property
    def rel_height(self):
        return self._rel_height

    @property
    def rel_width(self):
        return self._rel_width

    @property
    def rel_bottom(self):
        return self._rel_bottom

    @property
    def rel_right(self):
        return self._rel_right


class BoundingBoxAnnotationResult:
    """Data class to parse the result field saved by a SageMaker Ground Truth Object Detection job"""

    def __init__(self, manifest_obj: dict):
        try:
            image_size_spec = manifest_obj["image_size"][0]
            self._image_height = int(image_size_spec["height"])
            self._image_width = int(image_size_spec["width"])
            self._image_depth = (
                int(image_size_spec["depth"]) if "depth" in image_size_spec else None
            )
        except Exception as e:
            raise ValueError(
                "".join(
                    (
                        "manifest_obj must be a dictionary including 'image_size': a list of length 1 ",
                        "whose first/only element is a dict with integer properties 'height' and ",
                        f"'width', optionally also 'depth'. Got: {manifest_obj}",
                    )
                )
            ) from e
        assert (
            len(manifest_obj["image_size"]) == 1
        ), f"manifest_obj['image_size'] must be a list of len 1. Got: {manifest_obj['image_size']}"

        try:
            self._boxes = [
                AnnotationBoundingBox(
                    b,
                    image_height=self._image_height,
                    image_width=self._image_width,
                )
                for b in manifest_obj["annotations"]
            ]
        except Exception as e:
            raise ValueError(
                "".join(
                    (
                        "manifest_obj['annotations'] must be a list-like of absolute TLHW bounding box ",
                        f"dicts with class_id. Got {manifest_obj['annotations']}",
                    )
                )
            ) from e

    @property
    def image_height(self):
        return self._image_height

    @property
    def image_width(self):
        return self._image_width

    @property
    def image_depth(self):
        return self._image_depth

    @property
    def boxes(self):
        return self._boxes


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
    """Calculate overlap of Textract (TRP) items with SMGT Bounding Boxes to label word classes

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


class TextractLayoutLMDataset(Dataset):
    """PyTorch dataset for LayoutLM token classification from Textract + SMGT bounding boxes

    This dataset operates on JSONLines manifest files with records in the following form:

    record["textract-ref"] : str
        S3 URI to a response JSON from Amazon Textract
    record["page-num"] : int (optional)
        *1-based* page number of the Textract document that this record refers to
    record[annnotation_attr] : Dict
        An annotation record in similar format to what would be generated by a SageMaker Ground
        Truth Object Detection job, as required to parse a BoundingBoxAnnotationResult
    """

    def __init__(
        self,
        manifest_path: str,
        textract_channel: str,
        tokenizer,
        num_labels: int,
        annotation_attr: str,
        max_seq_len: int = 512,
        textract_prefix: str = "",
        pad_token_label_id=CrossEntropyLoss().ignore_index,
        cls_token_box=[0, 0, 0, 0],
        sep_token_box=[1000, 1000, 1000, 1000],
        pad_token_box=[0, 0, 0, 0],
    ):
        """Constructor for TextractLayoutLMDataset"""
        if not textract_channel.endswith("/"):
            textract_channel = textract_channel + "/"

        self.manifest_path = manifest_path
        self.textract_channel = textract_channel
        self.max_seq_len = max_seq_len
        self.textract_prefix = textract_prefix

        self._data = []

        # Examples start with [CLS] and end with [SEP], so there's some fixed overhead in how long
        # each example's content can be vs the raw max_seq_len:
        max_content_seq_len = max_seq_len - 2

        # Load the raw manifest data:
        with open(manifest_path, "r") as f:
            for ipagespec, manifest_line in enumerate(f, start=1):
                logger.debug(f"Reading manifest line {ipagespec}")
                pagespec = json.loads(manifest_line)
                if "textract-ref" not in pagespec:
                    raise ValueError(
                        f"Manifest line {ipagespec} missing required field 'textract-ref'"
                    )
                textract_path = pagespec["textract-ref"]
                if textract_path.lower().startswith("s3://"):
                    textract_s3key = textract_path[len("s3://") :].partition("/")[2]
                    if not textract_s3key.startswith(textract_prefix):
                        raise ValueError(
                            "".join(
                                (
                                    "(Manifest line {}) textract-ref {} S3 key does not start ",
                                    "with provided textract_prefix '{}'",
                                )
                            ).format(
                                ipagespec,
                                textract_path,
                                textract_prefix,
                            )
                        )
                    textract_relpath = textract_s3key[len(textract_prefix) :]
                    if textract_relpath.startswith("/"):
                        textract_path = textract_channel + textract_relpath[1:]
                    else:
                        textract_path = textract_channel + textract_relpath
                else:  # textract_path in manifest is not an S3 URI - assume relative to channel.
                    if textract_path.startswith("/"):
                        textract_path = textract_channel + textract_path[1:]
                    else:
                        textract_path = textract_channel + textract_path
                try:
                    with open(textract_path, "r") as ftextract:
                        textract_doc = trp.Document(json.loads(ftextract.read()))
                except FileNotFoundError:
                    raise ValueError(
                        "(Manifest line {}) could not find textract file {}".format(
                            ipagespec,
                            textract_path,
                        )
                    )
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
                if annotation_attr is not None:
                    if annotation_attr not in pagespec:
                        raise ValueError(
                            "(Manifest line {}) annotation_attr '{}' is missing".format(
                                ipagespec,
                                annotation_attr,
                            )
                        )
                    annotation = BoundingBoxAnnotationResult(pagespec[annotation_attr])
                else:
                    annotation = None
                tokenized = TextractLayoutLMDataset.tokenize_textract_doc(
                    textract_doc,
                    tokenizer,
                    max_seq_len=max_seq_len,
                    pad_token_label_id=pad_token_label_id,
                    cls_token_box=cls_token_box,
                    sep_token_box=sep_token_box,
                    pad_token_box=pad_token_box,
                    page_number=pagespec.get("page-num", 1),
                    return_tensors="pt",
                    annotation=annotation,
                    num_labels=num_labels,
                )
                logger.debug(
                    "Example {} generated: {}".format(
                        ipagespec,
                        "; ".join(
                            [
                                "'{}' ({})".format(
                                    k,
                                    getattr(tokenized[k], "shape", None)
                                    or (f"len {len(tokenized[k])}"),
                                )
                                for k in tokenized
                            ]
                        ),
                    )
                )
                if len(tokenized["input_ids"]) > 1:
                    logger.info(
                        "Example {} generated {} sequences: {}".format(
                            ipagespec,
                            len(tokenized["input_ids"]),
                            textract_path,
                        )
                    )
                # The tokenization already outputs a batch dimension because a single page may
                # produce multiple examples, so we need to distribute that batch dim back to our
                # dataset dim:
                for i in range(len(tokenized["input_ids"])):
                    self._data.append(
                        {
                            k: v[i]
                            for (k, v) in tokenized.items()
                            if k not in ("page_num", "textract_block_ids")
                        }
                    )
            # endfor line in manifest
        # endwith manifest file

    @staticmethod
    def tokenize_textract_doc(
        doc: trp.Document,
        tokenizer,
        max_seq_len: int = 512,
        pad_token_label_id=CrossEntropyLoss().ignore_index,
        cls_token_box=[0, 0, 0, 0],
        sep_token_box=[1000, 1000, 1000, 1000],
        pad_token_box=[0, 0, 0, 0],
        page_number=None,
        return_tensors=False,
        annotation=None,
        num_labels=None,
    ):
        """Tokenize a Textract document/page (perhaps with annotations) to model input

        Although the core transformers.LayoutLMTokenizer partially supports this task, we need to
        tokenize in such a way that:

        - Accurate position embeddings are included
        - Tokenized results can be linked back to Textract blocks

        Returns
        -------
        results : Dict
            Mapping names to arrays/tensors as expected by LayoutLM model: 'input_ids' (token IDs
            per tokenizer for the input); 'bbox' (normalized token bounding boxes);
            'attention_mask' (0 for padding tokens to max_seq_len and 1 otherwise);
            'token_type_ids' (1 for CLS and input word tokens, 0 for SEP and PAD)... And some
            additional outputs for linking outputs back to Textract: 'page_num' (a single 1-based
            page number per generated example in the batch); 'textract_block_ids' (string block
            UIDs for each token that maps to a Textract WORD block, else None for special tokens
            like CLS, SEP, PAD).
        """
        # Examples start with [CLS] and end with [SEP], so there's some fixed overhead in how long
        # each example's content can be vs the raw max_seq_len:
        max_content_seq_len = max_seq_len - 2

        if (annotation is not None) and (num_labels is None):
            raise ValueError(f"'num_labels' must be provided when 'annotation' is given")
        results = {
            # Actual LayoutLM input variables:
            "input_ids": [],
            "bbox": [],
            "attention_mask": [],
            "token_type_ids": [],
            # Additionals info for tracing from output back to Textract:
            "page_num": [],
            "textract_block_ids": [],
        }
        if annotation is not None:
            results["labels"] = []

        for page_num, page in (
            ((page_number, doc.pages[page_number - 1]),)
            if page_number is not None
            else enumerate(doc.pages, start=1)
        ):
            textract_words = [word for line in page.lines for word in line.words]
            if len(textract_words) == 0:
                # No words, skip this page
                # If you try to process, you'll get:
                # IndexError: too many indices for array: array is 1-dimensional, but 2 were
                # indexed at line:
                # token_boxes = word_boxes[tokenizing_indexes, :])
                continue
            word_block_ids = [word.id for word in textract_words]
            tokens_by_word = [tokenizer.tokenize(word.text) for word in textract_words]
            n_tokens_per_word = [len(ts) for ts in tokens_by_word]
            word_boxes = get_layoutlm_bounding_boxes(textract_words)

            if annotation is not None:
                word_labels = word_label_matrix_from_bounding_boxes(
                    textract_words,
                    annotation,
                    num_labels,
                )
                word_labels = np.where(
                    np.sum(word_labels, axis=1) == 0,
                    num_labels - 1,
                    np.argmax(word_labels, axis=1),
                )

            # Calculate indexes to slice word-wise lists into token-wise lists:
            tokenizing_indexes = [[i] * len(t) for i, t in enumerate(tokens_by_word)]
            tokenizing_indexes = [i for sublist in tokenizing_indexes for i in sublist]
            # Tokenize:
            tokens = [t for word_tokens in tokens_by_word for t in word_tokens]
            token_boxes = word_boxes[tokenizing_indexes, :]
            token_block_ids = [word_block_ids[i] for i in tokenizing_indexes]  # (not a np array)
            if annotation is not None:
                token_labels = word_labels[tokenizing_indexes]

            # When a page has more than our max_content_seq_len tokens, we split it into multiple
            # training examples:
            n_page_tokens = len(tokens)
            n_page_seqs = int(np.ceil(n_page_tokens / max_content_seq_len))
            for ixseq in range(n_page_seqs):
                # TODO: Better splitting strategy?
                seq_slice = slice(
                    (ixseq * max_content_seq_len), (max_content_seq_len * (ixseq + 1))
                )
                seq_tokens = tokens[seq_slice]
                seq_boxes = token_boxes[seq_slice, :].tolist()
                seq_block_ids = token_block_ids[seq_slice]

                # Append SEP token:
                seq_tokens += [tokenizer.sep_token]
                seq_boxes += [sep_token_box]
                seq_block_ids += [None]
                segment_ids = [0] * len(seq_tokens)

                # Prepend CLS token:
                seq_tokens = [tokenizer.cls_token] + seq_tokens
                seq_boxes = [cls_token_box] + seq_boxes
                seq_block_ids = [None] + seq_block_ids
                segment_ids = [1] + segment_ids

                seq_token_ids = tokenizer.convert_tokens_to_ids(seq_tokens)

                # The mask has 1 for 'real' tokens (including control) and 0 for padding tokens.
                # Only real tokens are attended to.
                input_mask = [1] * len(seq_token_ids)

                # Zero-pad up to the sequence length.
                padding_length = max_seq_len - len(seq_token_ids)
                seq_token_ids += [tokenizer.pad_token_id] * padding_length
                input_mask += [0] * padding_length
                segment_ids += [tokenizer.pad_token_id] * padding_length
                seq_boxes += [pad_token_box] * padding_length
                seq_block_ids += [None] * padding_length

                results["input_ids"].append(seq_token_ids)
                results["bbox"].append(seq_boxes)
                results["attention_mask"].append(input_mask)
                results["token_type_ids"].append(segment_ids)
                results["page_num"].append(page_num)
                results["textract_block_ids"].append(seq_block_ids)
                if annotation is not None:
                    seq_labels = token_labels[seq_slice].tolist()
                    seq_labels += [pad_token_label_id]  # Append SEP
                    seq_labels = [pad_token_label_id] + seq_labels  # Prepend CLS
                    seq_labels += [pad_token_label_id] * padding_length  # Append PAD
                    results["labels"].append(seq_labels)

        if return_tensors in (True, "pt"):
            for key in (k for k in results if k not in ("page_num", "textract_block_ids")):
                results[key] = torch.tensor(results[key], dtype=torch.long)
        elif return_tensors:
            raise ValueError("'return_tensors' only supports True or 'pt' (PyTorch)")
        return results

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


def get_dataset(channel: str, tokenizer, args, pad_token_label_id) -> TextractLayoutLMDataset:
    """Load a SMGT dataset from file/folder `channel`"""
    if os.path.isdir(channel):
        contents = os.listdir(channel)
        if len(contents) == 1:
            data_path = os.path.join(channel, contents[0])
        else:
            json_contents = list(
                filter(lambda s: re.search(r"\.jsonl?$", s), map(lambda s: s.lower(), contents))
            )
            if len(json_contents) == 1:
                data_path = os.path.join(channel, json_contents[0])
            else:
                raise ValueError(
                    "".join(
                        (
                            "Channel folder {} must contain exactly one file or exactly one .json/",
                            ".jsonl ...Got {}",
                        )
                    ).format(channel, contents)
                )
    elif os.path.isfile(channel):
        data_path = channel
    else:
        raise ValueError(f"Channel {channel} is neither file nor directory")

    return TextractLayoutLMDataset(
        data_path,
        textract_channel=args.textract,
        tokenizer=tokenizer,
        num_labels=args.num_labels,
        annotation_attr=args.annotation_attr,
        max_seq_len=args.max_seq_length,
        textract_prefix=args.textract_prefix,
        pad_token_label_id=pad_token_label_id,
    )


@dataclass
class DummyDataCollator:
    """Data collator that just stacks tensors from inputs.

    (We've already done the leg-work in our dataset class)
    """

    def __call__(self, features):
        return {k: torch.stack([f[k] for f in features]) for k in features[0]}
