# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""SageMaker inference wrapper for Amazon Textract LayoutLM Word Classification

API Usage
---------

All requests and responses in 'application/json'. This model takes Textract response-like 
(Amazon-Textract-Response-Parser compatible) JSON as input, and annotates each `WORD` block with
'ClassificationProbabilities', 'PredictedClass' and 'PredictedClassConfidence' properties.

Additionally, you can request with:

- req.S3Input : If the model execution role has sufficient permissions, set this property to a dict
  with key 'URI' OR ('Bucket' and 'Key') to instead have the model fetch the input JSON from S3.
  This is useful if you need to process JSONs bigger than the payload size limit (5MB by default).
- req.S3Output : Like S3Input, store the result to S3 and return a dict of { URI, Bucket, Key }
  instead of the full result inline.
- req.TargetPageNum : Set to a (1-based) integer if you only need to annotate a particular page of
  the input document.
- req.TargetPageOnly : Set true to output *only* the `TargetPageNum` if set - i.e. remove all other
  pages of the Textract JSON in the result.
- req.Content : Want to send your input Textract JSON inline, and also use some of the other
  parameters above - but don't like the idea of mixing the two together? No problem, just send the
  Textract JSON in the 'Content' key.
"""

# Python Built-Ins:
from collections import defaultdict
import io
import json
from operator import itemgetter
import os
from typing import Optional

# External Dependencies:
import boto3
import numpy as np

# Sadly (at transformers v4.6), LayoutLMTokenizer doesn't seem to work with AutoTokenizer as it
# expects config.json, not tokenizer_config.json:
from transformers import AutoConfig, LayoutLMTokenizerFast, AutoModelForTokenClassification
import torch
import trp

# Local Dependencies:
from .data.base import NaiveExampleSplitter
from .data.geometry import layoutlm_boxes_from_trp_blocks
from .data.ner import (
    TextractLayoutLMDataCollatorForWordClassification,
    TextractLayoutLMExampleForWordClassification,
)
from . import logging_utils

logger = logging_utils.getLogger("infcustom")
logger.info("Loading custom inference handlers")
# If you need to debug this script and aren't seeing any logging in CloudWatch, try setting the
# following on the Model to force flushing log calls through: env={ "PYTHONUNBUFFERED": "1" }


INFERENCE_BATCH_SIZE = int(os.environ.get("INFERENCE_BATCH_SIZE", "8"))
s3client = boto3.client("s3")


class S3ObjectSpec:
    """Utility class for parsing an S3 location spec from a JSON-able dict"""

    def __init__(self, spec: dict):
        if "URI" in spec:
            if not spec["URI"].lower().startswith("s3://"):
                raise ValueError("URI must be a valid 's3://...' URI if provided")
            bucket, _, key = spec["URI"][len("s3://") :].partition("/")
        else:
            bucket = spec.get("Bucket")
            key = spec.get("Key")
        if not (bucket and key and isinstance(bucket, str) and isinstance(key, str)):
            raise ValueError(
                "Must provide an object with either 'URI' or 'Bucket' and 'Key' properties. "
                f"Parsed bucket={bucket}, key={key}"
            )
        self.bucket = bucket
        self.key = key


def extract_textract_page(doc_json: dict, page_num: int, trp_doc: Optional[trp.Document] = None):
    """Extract just `page_num`'s data from a Textract JSON, using Textract-Response-Parser

    Arguments
    ---------
    doc_json : dict
        A Textract response-like JSON structure.
    page_num : int
        The (one-based) page number to extract
    trp_doc : trp.Document, optional
        Loaded TRP Document for doc_json, if you have one alreaady, to save cycles

    Returns
    -------
    doc_json : dict
        The reduced, Textract-like JSON structure (DocumentMetadata.Pages will be set =1)
    trp_doc : trp.Document
        Loaded TRP Document of doc_json
    """
    if not trp_doc:
        trp_doc = trp.Document(doc_json)
    result_json = {
        "Blocks": trp_doc.pages[page_num - 1].blocks,
        "DocumentMetadata": {
            "Pages": 1,
        },
    }
    for prop in ("DetectDocumentTextModelVersion",):
        if prop in doc_json:
            result_json[prop] = doc_json[prop]
    return {"doc_json": result_json, "trp_doc": trp.Document(result_json)}


def input_fn(input_bytes, content_type: str):
    """Deserialize and pre-process model request JSON

    Requests must be of type application/json. See module-level docstring for API details.
    """
    logger.info(f"Received request of type:{content_type}")
    if content_type != "application/json":
        raise ValueError("Content type must be application/json")

    req_json = json.loads(input_bytes)

    s3_input = req_json.get("S3Input")
    if s3_input:
        try:
            s3_input = S3ObjectSpec(s3_input)
        except ValueError as e:
            raise ValueError(
                "Invalid Request.S3Input: If provided, must be an object with 'URI' or 'Bucket' "
                "and 'Key'"
            ) from e
        logger.info(f"Fetching S3Input from s3://{s3_input.bucket}/{s3_input.key}")
        doc_json = json.loads(
            s3client.get_object(Bucket=s3_input.bucket, Key=s3_input.key)["Body"].read()
        )
        req_root_is_doc = False
    else:
        if "Content" in req_json:
            doc_json = req_json["Content"]
            req_root_is_doc = False
        else:
            doc_json = req_json
            req_root_is_doc = True

    s3_output = req_json.get("S3Output")
    if s3_output:
        try:
            s3_output = S3ObjectSpec(s3_output)
        except ValueError as e:
            raise ValueError(
                "Invalid Request.S3Output: If provided, must be an object with 'URI' or 'Bucket' "
                "and 'Key'"
            ) from e
        if req_root_is_doc:
            del doc_json["S3Output"]

    page_num = req_json.get("TargetPageNum")
    if page_num is not None:
        if req_root_is_doc:
            del doc_json["TargetPageNum"]

    target_page_only = req_json.get("TargetPageOnly")
    if target_page_only is not None:
        if req_root_is_doc:
            del doc_json["TargetPageOnly"]

    return {
        "doc_json": doc_json,
        "page_num": page_num,
        "s3_output": s3_output,
        "target_page_only": target_page_only,
    }


def output_fn(prediction_output, accept):
    """Finalize model result JSON.

    Requests must accept content type application/json. See module-level docstring for API details.
    """
    if accept != "application/json":
        raise ValueError("Accept header must be 'application/json'")

    doc_json, s3_output = itemgetter("doc_json", "s3_output")(prediction_output)

    if s3_output:
        logger.info(f"Uploading S3Output to s3://{s3_output.bucket}/{s3_output.key}")
        s3client.upload_fileobj(
            io.BytesIO(json.dumps(doc_json).encode("utf-8")),
            Bucket=s3_output.bucket,
            Key=s3_output.key,
        )
        return json.dumps(
            {
                "Bucket": s3_output.bucket,
                "Key": s3_output.key,
                "URI": f"s3://{s3_output.bucket}/{s3_output.key}",
            }
        ).encode("utf-8")
    else:
        return json.dumps(doc_json).encode("utf-8")


def model_fn(model_dir):
    """Load model artifacts from model_dir

    Returns
    -------
    collator : transformers.DataCollatorMixin
        Callable collator to prepare a batch of examples for model inference
    config : transformers.AutoConfig
        Would be useful if we could use transformers.pipeline, but sadly cannot
    model : transformers.AutoModelForTokenClassification
        Core HuggingFace Transformers model, initialized for evaluation and loaded to GPU if present
    tokenizer : transformers.LayoutLMTokenizerFast
        Core HuggingFace Tokenizer for LayoutLM as serialized in the model.pth
    device : torch.device
        Indicating which device the model was loaded to
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = LayoutLMTokenizerFast.from_pretrained(model_dir)
    collator = TextractLayoutLMDataCollatorForWordClassification(
        tokenizer,
        pad_to_multiple_of=8,
    )
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    model.to(device)
    config = AutoConfig.from_pretrained(model_dir)
    logger.info(f"Model loaded")
    return {
        "collator": collator,
        "config": config,
        "device": device,
        "model": model,
        "tokenizer": tokenizer,
    }


def predict_fn(input_data: dict, model: dict):
    """Classify WORD blocks on a Textract result using a LayoutLMForTokenClassification model

    Parameters
    ----------
    input_data : { doc_json, page_num, s3_output, target_page_only }
        Parsed JSON of Textract result, plus additional control parameters.
    model : { config, device, model, tokenizer }
        The core token classification model, tokenizer, config (not used) and PyTorch device.

    Returns
    -------
    doc_json : Union[List, Dict]
        Input Textract JSON with WORD blocks annotated with additional properties describing their
        classification according to the model: PredictedClass (integer ID of highest-scoring
        class), ClassificationProbabilities (list of floats scoring confidence for each possible
        class), and PredictedClassConfidence (float confidence of highest-scoring class).
    s3_output : S3ObjectSpec
        Passed through from input_data
    """
    collator, config, device, trained_model, tokenizer = itemgetter(
        "collator", "config", "device", "model", "tokenizer"
    )(model)
    doc_json, page_num, s3_output, target_page_only = itemgetter(
        "doc_json", "page_num", "s3_output", "target_page_only"
    )(input_data)
    trp_doc = trp.Document(doc_json)

    # Save memory by extracting individual page, if that's acceptable per the request:
    if target_page_only and page_num is not None:
        doc_json, trp_doc = itemgetter("doc_json", "trp_doc")(
            extract_textract_page(doc_json, page_num, trp_doc)
        )
        page_num = 1

    # We can't use pipeline/TextClassificationPipeline, because LayoutLMForTokenClassification has
    # been implemented such that the bbox input is separate and *optional*, and doesn't come from
    # the tokenizer!
    # So instead the logic here is heavily inspired by the pipeline but with some customizations:
    # https://github.com/huggingface/transformers/blob/f51188cbe74195c14c5b3e2e8f10c2f435f9751a/src/transformers/pipelines/token_classification.py#L115
    # nlp = pipeline(
    #     task="token-classification",
    #     model=trained_model,
    #     config=config,
    #     tokenizer=tokenizer,
    #     framework="pt",
    # )
    with torch.no_grad():
        # Split the page(s) into sequences of acceptable length for inference:
        examples = []
        example_word_block_ids = []
        for page in trp_doc.pages:
            page_words = [word for line in page.lines for word in line.words]
            page_word_texts = [word.text for word in page_words]
            page_word_boxes = layoutlm_boxes_from_trp_blocks(page_words)
            splits = NaiveExampleSplitter.split(
                [word.text for word in page_words],
                tokenizer,
                max_content_seq_len=config.max_position_embeddings - 2,
            )
            for startword, endword in splits:
                examples.append(
                    TextractLayoutLMExampleForWordClassification(
                        word_boxes_normalized=page_word_boxes[startword:endword],
                        word_texts=page_word_texts[startword:endword],
                    )
                )
                example_word_block_ids.append([word.id for word in page_words[startword:endword]])

        # Iterate batches:
        block_results_map = defaultdict(list)
        for ixbatch, _ in enumerate(examples[::INFERENCE_BATCH_SIZE]):
            ixbatchstart = ixbatch * INFERENCE_BATCH_SIZE
            batch_examples = examples[ixbatchstart : (ixbatchstart + INFERENCE_BATCH_SIZE)]
            batch = collator(batch_examples)
            for name in batch:  # Collect batch tensors to same GPU/target device:
                batch[name] = batch[name].to(device)
            output = trained_model.forward(**batch)
            # output.logits is (batch_size, seq_len, n_labels)

            # Convert logits to probabilities and retrieve to numpy:
            output_probs = torch.nn.functional.softmax(output.logits, dim=-1)
            probs_cpu = output_probs.cpu() if output_probs.is_cuda else output_probs
            probs = probs_cpu.numpy()

            # Map (sub-word, token-level) predictions per Textract BLOCK:
            for ixoffset, _ in enumerate(batch_examples):
                word_block_ids = example_word_block_ids[ixbatchstart + ixoffset]
                word_ids = batch.word_ids(ixoffset)
                for ixtoken, ixword in enumerate(word_ids):
                    if ixword is not None:
                        block_results_map[word_block_ids[ixword]].append(
                            probs[ixoffset, ixtoken, :]
                        )

        # Aggregate per-block results and save to Textract JSON:
        for block_id in block_results_map:
            block = trp_doc.getBlockById(block_id)
            block_probs = np.mean(
                np.stack(block_results_map[block_id]),
                axis=0,
            )
            # Remember numpy dtypes may not be JSON serializable, so convert to native types:
            block["ClassificationProbabilities"] = block_probs.tolist()
            block["PredictedClass"] = int(np.argmax(block_probs))
            block["PredictedClassConfidence"] = float(block_probs[block["PredictedClass"]])

    return {"doc_json": doc_json, "s3_output": s3_output}
