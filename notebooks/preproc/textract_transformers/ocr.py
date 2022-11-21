# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Script to run open-source OCR engines in Amazon SageMaker
"""

# Python Built-Ins:
import argparse
from base64 import b64decode
import json
from logging import getLogger
from multiprocessing import cpu_count, Pool
import os
import time
from typing import Iterable, Optional, Tuple

# External Dependencies:
import boto3

# Local Dependencies
from . import ocr_engines
from .file_utils import ls_relpaths
from .image_utils import Document


logger = getLogger("ocr")
s3client = boto3.client("s3")

# Environment variable configurations:
# When running in SM Endpoint, we can't use the usual processing job command line argument pattern
# to configure these extra parameters - so instead configure via environment variables for both.
OCR_ENGINE = os.environ.get("OCR_ENGINE", "tesseract").lower()
OCR_DEFAULT_LANGUAGES = os.environ.get("OCR_DEFAULT_LANGUAGES", "eng").lower().split(",")
OCR_DEFAULT_DPI = int(os.environ.get("OCR_DEFAULT_DPI", "300"))


def model_fn(model_dir: str):
    """OCR Engine loader: Load the configured engine into memory ready to use"""
    return ocr_engines.get(OCR_ENGINE, OCR_DEFAULT_LANGUAGES)


def input_fn(input_bytes: bytes, content_type: str) -> Tuple[Document, Optional[Iterable[str]]]:
    """Deserialize real-time processing requests

    For binary data requests (image or document bytes), default settings will be used e.g.
    `OCR_DEFAULT_LANGUAGES`.

    For JSON requests, supported fields are:

    ```
    {
        "Document": {
            "Bytes": Base64-encoded inline document/image, OR:
            "S3Object": {
                "Bucket": S3 bucket name for raw document/image
                "Name": S3 object key
                "VersionId": Optional S3 object version ID
            }
        },
        "Languages": Optional List[str] override for OCR_DEFAULT_LANGUAGES language codes
    }
    ```

    Returns
    -------
    doc :
        Loaded `Document` from which page images may be accessed
    languages :
        Optional override list of language codes for OCR, otherwise None.
    """
    logger.debug("Deserializing request of content_type %s", content_type)
    if content_type == "application/json":
        # Indirected request with metadata (e.g. language codes and S3 pointer):
        req = json.loads(input_bytes)
        doc_spec = req.get("Document", {})
        if "Bytes" in doc_spec:
            doc_bytes = b64decode(doc_spec["Bytes"])
        elif "S3Object" in doc_spec:
            s3_spec = doc_spec["S3Object"]
            if not ("Bucket" in s3_spec and "Name" in s3_spec):
                raise ValueError(
                    "Document.S3Object must be an object with keys 'Bucket' and 'Name'. Got: %s"
                    % s3_spec
                )
            logger.info("Fetching s3://%s/%s ...", s3_spec["Bucket"], s3_spec["Name"])
            version_id = s3_spec.get("Version")
            resp = s3client.get_object(
                Bucket=s3_spec["Bucket"],
                Key=s3_spec["Name"],
                **({} if version_id is None else {"VersionId": version_id}),
            )
            doc_bytes = resp["Body"].read()
            content_type = resp["ContentType"]
        else:
            raise ValueError(
                "JSON requests must include 'Document' object containing either 'Bytes' or "
                "'S3Object'. Got %s" % req
            )

        languages = req.get("Languages")
    else:
        # Direct image/document request:
        doc_bytes = input_bytes
        languages = None

    return (
        Document(doc_bytes, ext_or_media_type=content_type, default_doc_dpi=OCR_DEFAULT_DPI),
        languages,
    )


def predict_fn(
    inputs: Tuple[Document, Optional[Iterable[str]]], engine: ocr_engines.BaseOCREngine
) -> dict:
    """Get OCR results for a single input document/image, for the requested language codes

    Returns
    -------
    result :
        JSON-serializable OCR result dictionary, of format roughly compatible with Amazon Textract
        DetectDocumentText result payload.
    """
    return engine.process(inputs[0], languages=inputs[1])


# No output_fn required as we will always use JSON which the default serializer supports
# def output_fn(prediction_output: Dict, accept: str) -> bytes:


def parse_args() -> argparse.Namespace:
    """Parse SageMaker OCR Processing Job (batch) CLI arguments to job parameters"""
    parser = argparse.ArgumentParser(
        description="OCR documents in batch using an alternative (non-Amazon-Textract) engine"
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
        default="/opt/ml/processing/output/ocr",
        help="Folder where Amazon Textract-compatible OCR results should be saved",
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
    """Batch job worker function to extract a document (used in a multiprocessing pool)

    File paths are mapped similar to this sample's Amazon Textract pipeline. For example:
    `{in_folder}/some/fld/filename.pdf` to `{out_folder}/some/fld/filename.pdf/consolidated.json`

    Parameters
    ----------
    inputs :
        Dictionary containing fields:
        - in_folder (str): Mandatory path to input documents folder
        - rel_filepath (str): Mandatory path relative to `in_folder`, to input document
        - out_folder (str): Mandatory path to OCR results output folder
        - wait (float): Optional number of seconds to wait before starting processing, to ensure
            system resources are not *fully* exhausted when running as many threads as CPU cores.
            (Which could cause health check problems) - Default 0.5.
    """
    time.sleep(inputs.get("wait", 0.5))
    in_path = os.path.join(inputs["in_folder"], inputs["rel_filepath"])
    doc = Document(
        in_path,
        # ext_or_media_type to be inferred from file path
        default_doc_dpi=OCR_DEFAULT_DPI,
        base_file_path=inputs["in_folder"],
    )
    engine = ocr_engines.get(OCR_ENGINE, OCR_DEFAULT_LANGUAGES)
    try:
        result = engine.process(doc, OCR_DEFAULT_LANGUAGES)
    except Exception as e:
        logger.error("Failed to process document %s", in_path)
        raise e
    out_path = os.path.join(inputs["out_folder"], inputs["rel_filepath"], "consolidated.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(json.dumps(result, indent=2))
    logger.info("Processed doc %s", in_path)


def main() -> None:
    """Main batch processing job entrypoint: Parse CLI+envvars and process docs in multiple workers"""
    args = parse_args()
    logger.info("Parsed job args: %s", args)

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
                        "rel_filepath": path,
                    }
                    for path in rel_filepaths_all
                ],
            )
        ):
            logger.info("Processed doc %s of %s", ix + 1, n_docs)
    logger.info("All done!")
