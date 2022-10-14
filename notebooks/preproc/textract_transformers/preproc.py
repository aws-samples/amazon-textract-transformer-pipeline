# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Script to prepare raw documents into SMGT/model-ready page images (in batch or real-time)

This script can be used in a SageMaker Processing job to prepare images for SageMaker Ground Truth
labelling in batch (with optional extra thumbnails output by specifying a ProcessingOutput with
source /opt/ml/processing/output/thumbnails).

It can also be deployed as a SageMaker Endpoint (for asynchronous inference, to accommodate large
payloads in request/response) for generating page thumbnail bundles on-the-fly.
"""

# Python Built-Ins:
import argparse
from contextlib import nullcontext
from io import BytesIO
from logging import getLogger
from multiprocessing import cpu_count, Pool
import os
from tempfile import TemporaryDirectory
import time
from typing import Any, Dict

# External Dependencies:
import numpy as np

# Local Dependencies:
from .file_utils import ls_relpaths
from .image_utils import Document, PDF_CONTENT_TYPES, resize_image, SINGLE_IMAGE_CONTENT_TYPES


logger = getLogger("preproc")


# Environment variable configurations:
# When running in SM Endpoint, we can't use the usual processing job command line argument pattern
# to configure these extra parameters - so instead configure via environment variables for both.
PREPROC_THUMBNAIL_SIZE = tuple(
    int(x) for x in os.environ.get("PREPROC_THUMBNAIL_SIZE", "224,224").split(",")
)
if len(PREPROC_THUMBNAIL_SIZE) == 1:
    PREPROC_THUMBNAIL_SIZE = PREPROC_THUMBNAIL_SIZE[0]
PREPROC_PDF_DPI = int(os.environ.get("PREPROC_PDF_DPI", "300"))
PREPROC_DEFAULT_SQUARE = os.environ.get("PREPROC_DEFAULT_SQUARE", "true").lower()
if PREPROC_DEFAULT_SQUARE in ("true", "t", "yes", "y", "1"):
    PREPROC_DEFAULT_SQUARE = True
elif PREPROC_DEFAULT_SQUARE in ("false", "f", "no", "n", "0"):
    PREPROC_DEFAULT_SQUARE = False
else:
    raise ValueError(
        "Environment variable PREPROC_DEFAULT_SQUARE should be 'true', 'false', or not set"
    )
PREPROC_LETTERBOX_COLOR = os.environ.get("PREPROC_LETTERBOX_COLOR")
if PREPROC_LETTERBOX_COLOR:
    PREPROC_LETTERBOX_COLOR = tuple(int(x) for x in PREPROC_LETTERBOX_COLOR.split(","))
PREPROC_MAX_SIZE = os.environ.get("PREPROC_MAX_SIZE")
PREPROC_MAX_SIZE = int(PREPROC_MAX_SIZE) if PREPROC_MAX_SIZE else None
PREPROC_PREFERRED_IMAGE_FORMAT = os.environ.get("PREPROC_PREFERRED_IMAGE_FORMAT", "png")


def model_fn(model_dir: str) -> Any:
    """Dummy model loader: There is no "model" for this image processing case

    So long as predict_fn is present (so the container doesn't try to use this as a PyTorch model),
    it doesn't really matter what we return here.
    """
    return lambda x: x


def input_fn(input_bytes: bytes, content_type: str) -> Document:
    """Deserialize real-time processing requests

    Requests should be binary data (image or document), and this endpoint should typically be
    deployed as async to accommodate potentially large payload sizes.

    Returns
    -------
    result :
        Dict with "type" (an extension e.g. pdf, png, jpg) and **either** "image" (single loaded
        PIL image) **or** "doc" (raw document bytes for multi-page formats).
    """
    logger.debug("Deserializing request of content_type %s", content_type)
    return Document(
        input_bytes,
        ext_or_media_type=content_type,
        default_doc_dpi=PREPROC_PDF_DPI,
    )


def predict_fn(doc: Document, model: Any):
    """Execute real-time processing requests

    Either resize an individual image, or run the full thumbnail extraction process for a document.
    Document processing is done in a temporary directory

    Returns
    -------
    result :
        A dict with either "image" (a single PIL image, for single-image requests) or "images" (a
        list of PIL images, for document format inputs)
    """
    with (
        TemporaryDirectory() if doc.media_type in PDF_CONTENT_TYPES else nullcontext()
    ) as workspace_folder:
        if workspace_folder:
            doc.set_workspace(workspace_folder)
        thumbs = [
            resize_image(
                page.image,
                size=PREPROC_THUMBNAIL_SIZE,
                default_square=PREPROC_DEFAULT_SQUARE,
                letterbox_color=PREPROC_LETTERBOX_COLOR,
                max_size=PREPROC_MAX_SIZE,
            )
            for page in doc.get_pages()
        ]
    return {"images": thumbs} if len(thumbs) > 1 else {"image": thumbs[0]}


def output_fn(prediction_output: Dict, accept: str) -> bytes:
    """Serialize results for real-time processing requests

    Image response 'Accept' types (e.g. image/png) are supported only for single-image requests.

    application/x-npy will return an *uncompressed* numpy array of either:
    - Pixel data for single-image type requests, or
    - PNG file bytestrings for document/multi-page type requests

    application/x-npz (preferred for multi-page documents) will return a *compressed* numpy archive
    including either:
    - "image": Pixel data for single-image type requests, or
    - "images": PNG file bytestrings for document/multi-page type requests
    """
    if accept in SINGLE_IMAGE_CONTENT_TYPES:
        logger.info("Preparing single-image response")
        if "image" in prediction_output:
            buffer = BytesIO()
            prediction_output["image"].save(buffer, format=SINGLE_IMAGE_CONTENT_TYPES[accept])
            return buffer.getvalue()
        else:
            raise ValueError(
                f"Requested content type {accept} can only be used for single-page images. "
                "Try application/x-npz for a compressed numpy array of PNG image bytes."
            )
    elif accept in ("application/x-npy", "application/x-npz"):
        is_npz = accept == "application/x-npz"
        logger.info("Preparing %snumpy response", "compressed " if is_npz else "")
        if "image" in prediction_output:
            arr = np.array(prediction_output["image"].convert("RGB"))
            buffer = BytesIO()
            if is_npz:
                np.savez_compressed(buffer, image=arr)
            else:
                np.save(buffer, arr)
            return buffer.getvalue()
        else:
            imgs = []
            for img in prediction_output["images"]:
                with BytesIO() as buffer:
                    img.save(buffer, format=PREPROC_PREFERRED_IMAGE_FORMAT)
                    imgs.append(buffer.getvalue())
            imgs = np.array(imgs)
            buffer = BytesIO()
            if is_npz:
                np.savez_compressed(buffer, images=imgs)
            else:
                np.save(buffer, imgs)
            return buffer.getvalue()
    else:
        raise ValueError(
            f"Requested content type {accept} not recognised. Use application/x-npz (compressed) "
            "or application/x-npy - or (for single images) a supported image/... content type."
        )


def parse_args():
    """Parse SageMaker image pre-processing batch job CLI arguments"""
    parser = argparse.ArgumentParser(
        description="Split and standardize documents into images for annotation and model training"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/opt/ml/processing/input/raw",
        help="Folder where raw input images/documents are stored",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/opt/ml/processing/output/imgs-clean",
        help="Folder where cleaned output images should be saved",
    )
    default_thumbs_path = "/opt/ml/processing/output/thumbnails"
    parser.add_argument(
        "--thumbnails",
        type=str,
        default=default_thumbs_path if os.path.isdir(default_thumbs_path) else None,
        help=(
            "(Optional) folder where resized thumbnail images should be saved. Defaults to "
            f"{default_thumbs_path} IF this path exists, so the functionality can be enabled "
            "just by configuring the output in a SM Processing Job."
        ),
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=cpu_count(),
        help="Number of worker processes to use for extraction (default #CPU cores)",
    )
    args = parser.parse_args()
    return args


def process_doc_in_worker(inputs: dict) -> None:
    time.sleep(inputs.get("wait", 0.5))
    doc = Document(
        os.path.join(inputs["in_folder"], inputs["rel_filepath"]),
        # ext_or_media_type to be inferred from file path
        default_doc_dpi=PREPROC_PDF_DPI,
        default_image_ext=PREPROC_PREFERRED_IMAGE_FORMAT,
        base_file_path=inputs["in_folder"],
    )
    doc.set_workspace(inputs["out_folder"], multi_res=False)
    for ixpage, page in enumerate(doc.get_pages()):
        if inputs["thumbs_folder"]:
            thumb_path = page.file_path.replace(inputs["out_folder"], inputs["thumbs_folder"])
            # On first page only (for performance), ensure the output folder exists:
            if ixpage == 0:
                os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
            resize_image(
                page.image,
                size=PREPROC_THUMBNAIL_SIZE,
                default_square=PREPROC_DEFAULT_SQUARE,
                letterbox_color=PREPROC_LETTERBOX_COLOR,
                max_size=PREPROC_MAX_SIZE,
            ).save(thumb_path)


def main() -> None:
    """Main batch processing job entrypoint: Parse CLI+envvars and process docs in multiple workers"""
    args = parse_args()
    logger.info("Parsed job args: %s", args)
    logger.info("Additional thumbnail output is %s", "ENABLED" if args.thumbnails else "DISABLED")

    logger.info("Reading raw files from %s", args.input)
    rel_filepaths_all = ls_relpaths(args.input)

    n_docs = len(rel_filepaths_all)
    logger.info("Processing %s files across %s processes", n_docs, args.n_workers)
    with Pool(args.n_workers) as pool:
        for ix, _ in enumerate(
            pool.imap_unordered(
                process_doc_in_worker,
                [
                    {
                        "in_folder": args.input,
                        "out_folder": args.output,
                        "thumbs_folder": args.thumbnails,
                        "rel_filepath": path,
                    }
                    for path in rel_filepaths_all
                ],
            )
        ):
            logger.info("Processed doc %s of %s", ix + 1, n_docs)
    logger.info("All done!")
