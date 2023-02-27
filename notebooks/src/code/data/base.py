# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Base/common task data utilities for Amazon Textract + LayoutLM

This module defines utilities common across the different task types (e.g. MLM, NER)
"""
# Python Built-Ins:
from dataclasses import dataclass
from inspect import signature
import json
from numbers import Real
import os
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Union

# External Dependencies:
import datasets
import numpy as np
from PIL import Image
from transformers import BatchEncoding
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction
from transformers.processing_utils import ProcessorMixin
import torch
import trp

# Local Dependencies:
from ..logging_utils import getLogger
from .geometry import layoutlm_boxes_from_trp_blocks
from .splitting import duplicate_batch_record, map_split_long_samples

logger = getLogger("data.base")


@dataclass
class TaskData:
    """Base data interface exposed by the different task types (MLM, NER, etc) to training scripts

    Each new task module should implement a method get_task(data_args, tokenizer) -> TaskData
    """

    train_dataset: datasets.Dataset
    data_collator: Optional[Callable] = None
    eval_dataset: Optional[datasets.Dataset] = None
    metric_computer: Optional[Callable[[EvalPrediction], Dict[str, Real]]] = None


def asset_s3uri_to_file_path(
    s3uri: str,
    s3_prefix: str,
    path_prefix: str,
    typename: str = "Asset",
) -> str:
    """Map an S3 URI from manifest to local file path, via prefix"""
    s3key = s3uri[len("s3://") :].partition("/")[2]
    if not s3key.startswith(s3_prefix):
        raise ValueError(
            "%s S3 URI %s object key does not start with provided "
            "prefix '%s'" % (typename, s3uri, s3_prefix)
        )
    relpath = s3key[len(s3_prefix) :]
    if relpath.startswith("/"):
        # Because os.path.join('anything', '/slash/prefixed') = '/slash/prefixed'
        relpath = relpath[1:]
    return os.path.join(path_prefix, relpath)


def normalize_asset_ref(
    asset_ref: str,
    s3_prefix: str,
    path_prefix: str,
    typename: str = "Asset",
    check_exists: bool = True,
    manifest_line: Optional[int] = None,
) -> str:
    """Map a manifest 'ref' (S3 URI or raw path) to local path; and optionally check it exists"""
    if asset_ref[:5].lower() == "s3://":
        # Map S3 URI to local path:
        asset_ref = asset_s3uri_to_file_path(
            asset_ref,
            s3_prefix=s3_prefix,
            path_prefix=path_prefix,
            typename=typename,
        )
    else:
        # ref in manifest isn't an S3 URI - assume rel to channel
        if not path_prefix.endswith("/"):
            path_prefix = path_prefix + "/"
        if asset_ref.startswith("/"):
            # Because os.path.join('anything', '/slash/prefixed') = '/slash/prefixed'
            asset_ref = path_prefix + asset_ref[1:]
        else:
            asset_ref = path_prefix + asset_ref

    if check_exists:
        # Check the resolved file path exists:
        if not os.path.isfile(asset_ref):
            raise ValueError(
                "{}Could not find {} file {}".format(
                    f"(Manifest line {manifest_line}) " if manifest_line else "",
                    typename,
                    asset_ref,
                )
            )

    return asset_ref


def looks_like_hf_dataset(folder: str) -> bool:
    """Check if a local folder looks like a HuggingFace `Dataset` from save_to_disk(), or not"""
    if not os.path.isfile(os.path.join(folder, "dataset_info.json")):
        logger.debug(
            "Folder missing dataset_info.json does not appear to be HF Dataset: %s",
            folder,
        )
        return False
    elif not os.path.isfile(os.path.join(folder, "state.json")):
        logger.debug(
            "Folder missing state.json does not appear to be HF Dataset: %s",
            folder,
        )
        return False
    else:
        logger.debug("Folder appears to be saved Hugging Face Dataset: %s", folder)
        return True


def find_images_from_textract_path(
    textract_file_path: str,
    images_path: str,
    textract_path: str,
) -> Dict[int, str]:
    """Try to locate (local) page image files from (local) Textract file path

    For self-supervised situations where a document-level manifest is provided along with an images
    channel, so we need to try and look up the locations of page thumbnail images by file name.

    Returns
    -------
    result :
        Dict from (one-based) page number to local thumbnail image path. May or may not be fully
        contiguous, but will have at least one entry (or a ValueError would be raised).
    """

    if not textract_file_path.startswith(textract_path):
        raise ValueError(
            "Couldn't find page images from Textract path: textract_file_path does not start with "
            f"textract_path: Got '{textract_file_path}', '{textract_path}"
        )
    textract_relpath = textract_file_path[len(textract_path) :]
    if textract_relpath.startswith("/"):
        textract_relpath = textract_relpath[1:]

    if textract_file_path.endswith("/consolidated.json"):
        # Expected case: path/to/myfile.pdf/consolidated.json
        doc_relpath = textract_relpath[: -len("/consolidated.json")]
    else:
        # Also supported: Textract JSON path is same as raw doc/image path
        if not textract_file_path.rpartition(".")[2] in (
            "jpg",
            "jpeg",
            "pdf",
            "png",
            "tif",
            "tiff",
        ):
            logger.warning(
                "Trying to find thumbnails from a path that doesn't look like either a Textracted "
                f"JSON (with /consolidated.json) or a raw .jpg/png/pdf/etc: {textract_file_path}"
            )
        doc_relpath = textract_relpath

    # Expected behaviour:
    # - path/to/mydoc.pdf -> path/to/mydoc-\d{4,}-\d+.(png|jpg|jpeg)
    # - path/to/myimg.png -> path/to/myimg.png
    # - path/to/mydoc.tiff -> path/to/mytiff-\d{4,}.(png|jpg|jpeg)
    doc_reldirname, _, doc_filename = doc_relpath.rpartition("/")
    doc_basename, _, doc_ext = doc_filename.rpartition(".")
    doc_ext = doc_ext.lower()
    images_basepath = os.path.join(images_path, doc_reldirname)
    if not (os.path.isdir(images_basepath)):
        raise ValueError(
            f"Couldn't find thumbnails from Textract path {textract_file_path}: Expected "
            f"{images_basepath} to be a folder"
        )
    if doc_ext != "pdf":
        candidate = os.path.join(images_basepath, doc_filename)
        if os.path.isfile(candidate):
            return {1: candidate}
    candidate_filenames = list(
        filter(lambda f: f.startswith(doc_basename), os.listdir(images_basepath))
    )

    # Extract page numbers and sort results by them (filtering out any that aren't as expected):
    nums_exp = re.compile(r"-(\d{4,}-)?(\d+)\.(png|jpg|jpeg)")
    doc_basename_len = len(doc_basename)
    candidate_matches = [nums_exp.match(f[doc_basename_len:]) for f in candidate_filenames]
    candidate_pagenums = [
        int(m.group(2)) if m is not None and len(m.groups()) >= 2 else None
        for m in candidate_matches
    ]

    result = {
        num: os.path.join(images_basepath, candidate_filenames[ix])
        for ix, num, in enumerate(candidate_pagenums)
        if num is not None
    }
    if not len(result):
        raise ValueError(
            f"Found no thumbnails matching {textract_file_path} in {images_path}: "
            f"Raw candidates {candidate_filenames}"
        )
    return result


def map_load_text_and_images(
    batch: Dict[str, List],
    idxs: List[Any],
    images_path: Optional[str] = None,
    images_prefix: str = "",
    textract_path: str = "",
    textract_prefix: str = "",
    # TODO: output_line_ids seems to be broken atm? At least for NER/seq2seq it's throwing errors
    output_line_ids: bool = False,
) -> Dict[str, List]:
    """datasets.map function to load examples for a manifest-file-like batch

    Input batches should have fields:
    - "textract-ref" (Required) - S3 URI of consolidated document Textract JSONs
    - "source-ref" (Optional) - S3 URI of page (thumbnail) image
    - "page-num" (Optional) - If the file is page-wise

    If "page-num" is not present in a record, the record will be expanded to include every page in
    the source document (*copying* any annotations - no splitting implemented). If "source-ref" is
    not included but `images_path` is provided, a lookup of the thumbnail images for the given
    document will be attempted (and raise an error if it fails).

    Note using "source-ref" as a thumbnail path works because the pre-processing job mirrors folder
    and filename structure between cleaned/SMGT images and thumbnail images.

    Output batches will be augmented with:
    - "text": List of word texts
    - "boxes": LayoutLM-normalized bounding boxes per word in "text"
    - "images": (Optional) numpy pixel array per image. **Note:** this will be converted to native
        PyArrow arrays by the `datasets` library, so may appear as plain nested lists in subsequent
        mappings, which LayoutLMv2Processor won't like (as it expects PIL images or np/pt tensors).
        Why do we return pixel arrays instead of PIL images? Because at the time of writing,
        returning PIL images caused deadlock when a custom data collator is used and
        `dataloader_num_workers > 0`. This was true even if an Image `.copy()` was returned by this
        function, and even if `splitting.duplicate_batch_record` and `splitting.split_batch_record`
        were also updated to .copy() duplicated fields where possible.
    - "line-ids": (If `output_line_ids` is True) List of 0-based indexes marking which Textract
        `LINE` block on the page each word in "text" is associated with. This is used for some data
        collation tasks that need to be aware of the grouping of words into contiguous lines.
    """
    if "textract-ref" not in batch:
        raise ValueError(f"Manifest batch at line {idxs[0]} missing required field 'textract-ref'")
    textract_refs_raw = batch["textract-ref"]
    missing_textract_refs = [i for i, x in enumerate(textract_refs_raw) if not x]
    if len(missing_textract_refs):
        raise ValueError(
            "Manifest line(s) missing required field 'textract-ref': {}".format(
                [idxs[i] for i in missing_textract_refs]
            )
        )
    if images_path:
        if "source-ref" not in batch:
            image_refs_raw = [None for _ in batch["textract-ref"]]
        else:
            image_refs_raw = batch["source-ref"]
    else:
        image_refs_raw = None

    # A page-wise manifest may have multiple entries for (different pages of) one textract-ref:
    textract_paths_by_ref: Dict[str, str] = {}
    # Let's also record which input items can be processed with same TRP doc in memory:
    batch_idxs_by_textract_path: Dict[str, List[int]] = {}
    for i, ref in enumerate(textract_refs_raw):
        if ref in textract_paths_by_ref:
            batch_idxs_by_textract_path[textract_paths_by_ref[ref]].append(i)
        else:
            textract_paths_by_ref[ref] = normalize_asset_ref(
                ref,
                s3_prefix=textract_prefix,
                path_prefix=textract_path,
                typename="Textract",
                check_exists=True,
                manifest_line=idxs[i],
            )
            batch_idxs_by_textract_path[textract_paths_by_ref[ref]] = [i]

    # Start shaping our output:
    input_n_records = len(idxs)
    input_to_output = list(range(input_n_records))
    output = {k: [val for val in v] for k, v in batch.items()}
    output["text"] = [None] * input_n_records
    output["boxes"] = [None] * input_n_records
    if output_line_ids:
        output["line-ids"] = [None] * input_n_records
    if "page-num" not in batch:
        output["page-num"] = [None] * input_n_records

    if image_refs_raw:
        image_paths = [
            normalize_asset_ref(
                img_ref,
                s3_prefix=images_prefix,
                path_prefix=images_path,
                typename="Image",
                check_exists=True,
                manifest_line=idxs[i],
            )
            if img_ref
            else None
            for img_ref in image_refs_raw
        ]
        output["images"] = [
            np.array(Image.open(path).convert("RGB")) if path else None for path in image_paths
        ]

    # Since TRP gives us no choice but to load a whole Textract doc at a time (even if manifest
    # items are individual pages), proceed by Textract document and batch up the affected manifest
    # items if more than 1:
    for textract_file_path, record_ixs in batch_idxs_by_textract_path.items():
        with open(textract_file_path, "r") as f:
            try:
                doc = trp.Document(json.loads(f.read()))
            except Exception as err:
                logger.exception(
                    "Couldn't parse Textract JSON file %s (used by manifest lines %s)",
                    textract_file_path,
                    [idxs[i] for i in record_ixs],
                )
                raise err
        auto_doc_images = None  # Only auto-discover doc thumbnails if required by some record

        for i in record_ixs:
            iout = input_to_output[i]
            page_num = None if "page-num" not in batch else batch["page-num"][i]
            if page_num is not None:
                # This record stays as just one record in the output batch:
                page = doc.pages[page_num - 1]
                words = [word for line in page.lines for word in line.words]
                word_boxes = layoutlm_boxes_from_trp_blocks(words)
                word_texts = [word.text for word in words]
                output["boxes"][iout] = word_boxes
                output["text"][iout] = word_texts
                if output_line_ids:
                    output["line-ids"] = [
                        ixline for ixline, line in enumerate(page.lines) for _ in line.words
                    ]
                if image_refs_raw and output["images"][iout] is None:
                    if not auto_doc_images:
                        auto_doc_images = find_images_from_textract_path(
                            textract_file_path,
                            images_path=images_path,
                            textract_path=textract_path,
                        )
                    if page_num not in auto_doc_images:
                        raise ValueError(
                            f"Couldn't find thumbnail for page {page_num} of Textract doc "
                            f"{textract_file_path}. Got: {auto_doc_images}"
                        )
                    output["images"][iout] = np.array(
                        Image.open(auto_doc_images[page_num]).convert("RGB")
                    )
            else:
                # This record becomes n_pages records in the output batch:
                doc_n_pages = len(doc.pages)
                words_by_page = [
                    [word for line in page.lines for word in line.words] for page in doc.pages
                ]
                if image_refs_raw:
                    if not auto_doc_images:
                        auto_doc_images = find_images_from_textract_path(
                            textract_file_path,
                            images_path=images_path,
                            textract_path=textract_path,
                        )
                    for ix in range(doc_n_pages):
                        if (ix + 1) not in auto_doc_images:
                            raise ValueError(
                                f"Couldn't find thumbnail for page {ix + 1} of Textract doc "
                                f"{textract_file_path}. Got: {auto_doc_images}"
                            )
                    images_by_page = [
                        np.array(Image.open(auto_doc_images[ix + 1]).convert("RGB"))
                        for ix in range(doc_n_pages)
                    ]
                else:
                    images_by_page = None
                extras = {}
                if images_by_page:
                    extras["images"] = images_by_page
                if output_line_ids:
                    extras["line-ids"] = [
                        [ixline for ixline, line in enumerate(page.lines) for _ in line.words]
                        for page in doc.pages
                    ]
                output = duplicate_batch_record(
                    output,
                    ix=iout,
                    n_copies=doc_n_pages,
                    feature_overrides={
                        "boxes": [
                            layoutlm_boxes_from_trp_blocks(page_words)
                            for page_words in words_by_page
                        ],
                        "text": [
                            [word.text for word in page_words] for page_words in words_by_page
                        ],
                        **extras,
                    },
                )
                input_to_output = input_to_output[0 : i + 1] + [
                    ix + doc_n_pages - 1 for ix in input_to_output[i + 1 :]
                ]

    # Drop any pages/examples that have no text content at all:
    has_content = [
        len(text) > 0 and len(output["boxes"][ix]) > 0 for ix, text in enumerate(output["text"])
    ]
    n_missing_content = len(has_content) - sum(has_content)
    if n_missing_content > 0:
        logger.info("Dropping %s samples with no text content", n_missing_content)
        output = {
            k: [v[ix] for ix, keep in enumerate(has_content) if keep] for k, v in output.items()
        }

    return output


def prepare_base_dataset(
    textract_path: str,
    manifest_file_path: Optional[str] = None,
    images_path: Optional[str] = None,
    images_prefix: str = "",
    textract_prefix: str = "",
    output_line_ids: bool = False,
    num_workers: Optional[int] = None,
    batch_size: int = 16,
    cache_dir: Optional[str] = None,
    map_cache_file_name: Optional[str] = None,
) -> datasets.Dataset:
    """Prepare a base datasets.Dataset **without** splitting long samples

    This basic preparation is common between task types, but does not yet apply
    `split_long_dataset_samples()` because you usually want to do any other processing before that.

    Parameters
    ----------
    textract_path :
        Local path to consolidated document Textract JSON files
    manifest_file_path :
        Optional path to manifest file describing the dataset (otherwise the full contents of
        `textract_path` will be used).
    images_path :
        Optional local path to resized/thumbnail page images
    images_prefix :
        S3 prefix under which "source-ref" URIs in the manifest file can be mapped to `images_path`
    textract_prefix :
        S3 prefix under which "textract-ref" URIs in the manifest file can be mapped to
        `textract_path`
    output_line_ids :
        Set True to augment dataset with the "line-ids" field listing 0-based contiguous line
        number for each word in "text" (to enable grouping words back into lines).
    num_workers :
        Number of parallel worker threads to use for loading the dataset.
    cache_dir :
        Folder to cache intermediate dataset PyArrow files in (ensure this is under a SageMaker EBS
        mount to avoid running out of space on the root device and failing the training job!)
    """
    if not os.path.isdir(textract_path):
        raise ValueError("textract_path '%s' is not a valid folder" % textract_path)
    if not textract_path.endswith("/"):
        textract_path = textract_path + "/"
    if images_path and not images_path.endswith("/"):
        images_path = images_path + "/"

    if manifest_file_path:
        if os.path.isfile(manifest_file_path):
            ds_raw = datasets.load_dataset(
                "json",
                data_files=manifest_file_path,
                split=datasets.Split.ALL,
                cache_dir=cache_dir,
            )
        else:
            if not os.path.isdir(manifest_file_path):
                raise ValueError(
                    f"Data manifest '{manifest_file_path}' is not a local file or folder"
                )
            # Fix for load_dataset() appearing to duplicate records in some cases:
            # Instead of loading all files in the folder with something like the below...
            #
            #   ds_raw = datasets.load_dataset(
            #       manifest_file_path,
            #       split=datasets.Split.ALL,
            #       cache_dir=cache_dir,
            #   )
            #
            # ...Explicitly filter out paths with '/.' in them to remove any stray
            # .ipynb_checkpoints folders that might have been loaded to S3 and our job:
            ds_raw = datasets.load_dataset(
                "json",
                data_files=[
                    os.path.join(currdir, f)
                    for currdir, _, files in os.walk(manifest_file_path)
                    for f in files
                    if "/." not in currdir
                ],
                split=datasets.Split.ALL,
                cache_dir=cache_dir,
            )
    else:
        ds_raw = datasets.Dataset.from_dict(
            {
                "textract-ref": [
                    # Output paths *relative* to textract_path:
                    os.path.join(os.path.relpath(currpath, textract_path), file)
                    for currpath, _, files in os.walk(textract_path)
                    for file in files
                ]
            },
            # At writing, from_dict() doesn't support setting cache_dir
        )

    if not datasets.utils.is_progress_bar_enabled():
        logger.info("Loading text and images... (progress bar disabled)")
    return ds_raw.map(
        map_load_text_and_images,
        with_indices=True,  # Only really used for error diagnostics
        batched=True,
        batch_size=batch_size,
        remove_columns=ds_raw.column_names,  # Strip all raw/original fields - only take fn returns
        fn_kwargs={
            "images_path": images_path,
            "images_prefix": images_prefix,
            "textract_path": textract_path,
            "textract_prefix": textract_prefix,
            "output_line_ids": output_line_ids,
        },
        num_proc=num_workers,
        desc="Loading text and images",
        cache_file_name=map_cache_file_name,
    )


def split_long_dataset_samples(
    dataset: datasets.Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
    batched: bool = True,
    batch_size: int = 16,
    desc: Optional[str] = "Splitting long pages",
    num_workers: Optional[int] = None,
    **other_map_kwargs,
) -> datasets.Dataset:
    """Transform a HF dataset by splitting samples longer than max_seq_len"""
    tokenizer_params = set(signature(tokenizer).parameters)

    fn_kwargs = {
        "tokenizer": tokenizer,
        "max_seq_len": max_seq_len,
        # Could consider exposing 'splitter' as an option here too but eh.
        "tokenizer_params": tokenizer_params,
    }
    # Try and prevent first-use state changes from changing tokenizer's hash result for datasets:
    # https://github.com/huggingface/transformers/issues/14931
    # TODO: Can this be removed once fixed or because we have explicit cache file names now?
    logger.info("Pre-warming tokenizer before split .map()")
    map_split_long_samples(
        dataset[0:1],
        **fn_kwargs,
    )
    logger.info("Tokenizer pre-warm done")

    if desc and not datasets.utils.is_progress_bar_enabled():
        logger.info("%s... (progress bar disabled)", desc)
    return dataset.map(
        map_split_long_samples,
        batched=batched,
        batch_size=batch_size,
        remove_columns=dataset.column_names,
        fn_kwargs=fn_kwargs,
        num_proc=num_workers,
        desc=desc,
        **other_map_kwargs,
    )


def get_tokenizer_extra_kwargs(
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: Optional[int] = None,
    pad_to_multiple_of: Optional[int] = 8,
    are_batches_final: bool = False,
    overrides: Optional[Dict[str, Any]] = None,
    tokenizer_param_names: Optional[Union[Mapping[str, Any], Set[str]]] = None,
) -> Dict[str, Any]:
    """Generate kwargs to configure tokenizers consistently between supported model versions

    Inspects the parameters of `tokenizer` and adds relevant settings depending on the version.

    Parameters
    ----------
    tokenizer :
        Tokenizer to be configured
    max_seq_len :
        Optional maximum number of tokens for the model
    pad_to_multiple_of :
        Optional padding configuration for the tokenizer
    are_batches_final :
        When tokenizing a dataset in advance, the processing batches are not the final model
        batches and therefore the only viable padding strategy (to ensure all model inputs in a
        training batch match) is "max_length". When `are_batches_final` is not True,
        `pad_to_multiple_of` will be ignored.
    overrides :
        Optional set of kwargs which will override these defaults
    tokenizer_param_names :
        Optional pre-calculated set of tokenizer parameter names (otherwise this function will
        inspect the tokenizer's signature)

    Returns
    -------
    kwargs :
        Additional configuration arguments to use your LayoutLMV1/V2/XLM tokenizer (or processor).
    """
    if not tokenizer_param_names:
        tokenizer_param_names = signature(tokenizer).parameters

    tokenizer_kwargs = {k: v for k, v in overrides.items()} if overrides else {}

    # Common setup:
    if max_seq_len is not None:
        tokenizer_kwargs["max_length"] = max_seq_len
    if not are_batches_final:
        if pad_to_multiple_of is not None:
            logger.warning(
                "Ignoring pad_to_multiple_of=%s because are_batches_final is False. Padding must "
                "be max_seq_len unless dealing with final training-ready batches.",
                pad_to_multiple_of,
            )
        if max_seq_len is not None:
            tokenizer_kwargs["padding"] = "max_length"
    elif pad_to_multiple_of is not None:
        tokenizer_kwargs["padding"] = bool(pad_to_multiple_of)
        tokenizer_kwargs["pad_to_multiple_of"] = pad_to_multiple_of

    # Automatic handling of different LayoutLM-based tokenizer versions:
    if ("is_split_into_words" in tokenizer_param_names) and (
        "is_split_into_words" not in tokenizer_kwargs
    ):
        tokenizer_kwargs["is_split_into_words"] = True
    if ("return_attention_mask" in tokenizer_param_names) and (
        "return_attention_mask" not in tokenizer_kwargs
    ):
        tokenizer_kwargs["return_attention_mask"] = True
    if ("return_token_type_ids" in tokenizer_param_names) and (
        "return_token_type_ids" not in tokenizer_kwargs
    ):
        tokenizer_kwargs["return_token_type_ids"] = True

    return tokenizer_kwargs


@dataclass
class LayoutLMDataCollatorMixin:
    """A data collator mixin to handle the weirdnesses of multi-modal/LayoutLM models

    Call _init_for_layoutlm() somewhere in your __post_init__ to set up.
    """

    # Special box values for special tokens:
    bos_token_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    pad_token_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    sep_token_box: Tuple[int, int, int, int] = (1000, 1000, 1000, 1000)
    processor: Optional[ProcessorMixin] = None

    def _init_for_layoutlm(self):
        self.special_token_boxes = torch.LongTensor(
            [
                self.bos_token_box,
                self.pad_token_box,
                self.sep_token_box,
            ]
        )
        # Add extra parameters depending what this tokenizer needs:
        # (Just store simple set of names rather than whole parameters object to try and avoid any
        # multithread problems that might hide in there)
        self.tokenizer_param_names = set(signature(self.tokenizer).parameters)
        self.tokenizer_extra_kwargs = get_tokenizer_extra_kwargs(
            self.tokenizer,
            max_seq_len=None,
            pad_to_multiple_of=self.pad_to_multiple_of,
            are_batches_final=True,
            overrides={},
            tokenizer_param_names=self.tokenizer_param_names,
        )
        self.processor_param_names = (
            set(signature(self.processor).parameters) if self.processor else None
        )

    def _map_sample_line_ids(
        self,
        word_line_ids: List[int],
        token_word_ids: List[Union[int, None]],
    ) -> torch.LongTensor:
        """Map one sequence's word-level line IDs to token-level.

        Returns
        -------
        result :
            (seq_len,) tensor of line ID numbers for each token in the sequence, or -1 where the
            token does not correspond to real text (e.g. special CLS, SEP, PAD tokens, etc.)
        """
        n_words = len(word_line_ids)
        augmented_example_line_ids = torch.LongTensor(word_line_ids + [-1])
        # Torch tensors don't support None->NaN, but numpy float ndarrays do:
        line_ids_np = np.nan_to_num(np.array(token_word_ids, dtype=float), nan=n_words)
        return torch.index_select(
            augmented_example_line_ids,
            0,
            # By this point all NaNs from special tokens should be resolved so can cast:
            torch.LongTensor(line_ids_np.astype(int)),
        )

    def _map_word_boxes(
        self, tokenized: BatchEncoding, example_word_boxes: List[List[Tuple[int, int, int, int]]]
    ):
        """Map word bounding boxes onto `tokenized["bbox"]`, if necessary

        `bbox` is the forward() param for all our supported model versions (LLMv1, v2, XLM) but
        some versions' tokenizers don't automatically map input boxes through.
        """
        if "bbox" in tokenized:
            return tokenized  # No action required

        n_examples = tokenized.n_sequences
        n_example_words = [len(boxes) for boxes in example_word_boxes]
        bbox_tensors_by_example = []
        for ixex in range(n_examples):
            word_ids = tokenized.word_ids(ixex)
            n_words = n_example_words[ixex]

            augmented_example_word_boxes = torch.cat(
                (
                    torch.LongTensor(example_word_boxes[ixex]),
                    self.special_token_boxes,
                ),
                dim=0,
            )
            # Torch tensors don't support None->NaN, but numpy float ndarrays do:
            box_ids_np = np.array(word_ids, dtype=float)
            box_ids_np = np.where(
                tokenized.input_ids[ixex, :] == self.tokenizer.bos_token_id,
                n_words,  # bos_token_box, per special_token_boxes
                box_ids_np,
            )
            box_ids_np = np.where(
                tokenized.input_ids[ixex, :] == self.tokenizer.cls_token_id,
                n_words,  # cls_token_box, per special_token_boxes
                box_ids_np,
            )
            box_ids_np = np.where(
                tokenized.input_ids[ixex, :] == self.tokenizer.pad_token_id,
                n_words + 1,  # pad_token_box, per special_token_boxes
                box_ids_np,
            )
            box_ids_np = np.where(
                tokenized.input_ids[ixex, :] == self.tokenizer.sep_token_id,
                n_words + 2,  # sep_token_box, per special_token_boxes
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

        tokenized["bbox"] = torch.stack(bbox_tensors_by_example)
