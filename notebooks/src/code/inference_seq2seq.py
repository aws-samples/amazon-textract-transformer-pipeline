# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Alternative SageMaker inference wrapper for text-only (non-multimodal) seq2seq models

These models are optionally deployed alongside the core layout-aware NER model, to normalize
detected entity mentions.

API Usage
---------

All requests and responses in 'application/json'. The model takes a dict with key `inputs` which
may be a text string or a list of strings. It will return a dict with key `generated_text`
containing either a text string or a list of strings (as per the input).
"""

# Python Built-Ins:
import json
import os
from typing import Dict, List, Union

# External Dependencies:
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# Local Dependencies:
from . import logging_utils

logger = logging_utils.getLogger("infcustom")
logger.info("Loading custom inference handlers")
# If you need to debug this script and aren't seeing any logging in CloudWatch, try setting the
# following on the Model to force flushing log calls through: env={ "PYTHONUNBUFFERED": "1" }

# Configurations:
INFERENCE_BATCH_SIZE = int(os.environ.get("INFERENCE_BATCH_SIZE", "4"))
PAD_TO_MULTIPLE_OF = os.environ.get("PAD_TO_MULTIPLE_OF", "8")
PAD_TO_MULTIPLE_OF = None if PAD_TO_MULTIPLE_OF in ("None", "") else int(PAD_TO_MULTIPLE_OF)


def input_fn(input_bytes, content_type: str):
    """Deserialize and pre-process model request JSON

    Requests must be of type application/json. See module-level docstring for API details.
    """
    logger.info(f"Received request of type:{content_type}")
    if content_type != "application/json":
        raise ValueError("Content type must be application/json")

    req_json = json.loads(input_bytes)
    if "inputs" not in req_json:
        raise ValueError(
            "Request JSON must contain field 'inputs' with either a text string or a list of text "
            "strings"
        )
    return req_json["inputs"]


# No custom output_fn needed as result is plain JSON fully prepared by predict_fn


def model_fn(model_dir) -> dict:
    """Load model artifacts from model_dir into a dict

    Returns
    -------
    pipeline : transformers.pipeline
        HF Pipeline for text generation inference
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        pad_to_multiple_of=PAD_TO_MULTIPLE_OF,
        # TODO: Is it helpful to use_fast=True?
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.eval()
    model.to(device)

    pl = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=INFERENCE_BATCH_SIZE,
        # num_workers as per default
        device=model.device,
    )

    logger.info("Model loaded")
    return {
        # Could return other objects e.g. `model` and `tokenizer`` for debugging
        "pipeline": pl,
    }


def predict_fn(
    input_data: Union[str, List[str]],
    model_data: dict,
) -> Dict[str, Union[str, List[str]]]:
    """Generate text outputs from an input or list of inputs

    Parameters
    ----------
    input_data :
        Input text string or list of input text strings (including prompts if needed)
    model_data : { pipeline }
        Trained model data loaded by model_fn, including a `pipeline`.

    Returns
    -------
    result :
        Dict including key `generated_text`, which will either be a text string (if `input_data` was
        a single string) or a list of strings (if `input_data` was a list).
    """
    pl = model_data["pipeline"]

    # Use transformers Pipelines to simplify the inference process and handle e.g. batching and
    # tokenization for us:
    result = pl(input_data, clean_up_tokenization_spaces=True)

    # Convert output from list of dicts to dict of lists:
    result = {k: [r[k] for r in result] for k in result[0].keys()}
    # Strip any leading/trailing whitespace from results:
    result["generated_text"] = [t.strip() for t in result["generated_text"]]

    # If input was a plain string (instead of a list of strings), remove the batch dimension from
    # outputs too:
    if isinstance(input_data, str):
        for k in result:
            result[k] = result[k][0]

    return result
