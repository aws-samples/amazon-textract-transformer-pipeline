# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Validation/accuracy metric callbacks for seq2seq modelling tasks"""
# Python Built-Ins:
from numbers import Real
from typing import Callable, Dict

# External Dependencies:
import numpy as np
from transformers import EvalPrediction, PreTrainedTokenizerBase


def get_metric_computer(
    tokenizer: PreTrainedTokenizerBase,
) -> Callable[[EvalPrediction], Dict[str, Real]]:
    """An 'accuracy' computer for seq2seq tasks that ignores outer whitespace and case.

    For our example task, it's reasonable to measure exact-match accuracy (since we're normalising
    small text spans - not e.g. summarizing long texts to shorter paragraphs). Therefore this metric
    computer checks exact accuracy, while allowing for variations in case and leading/trailing
    whitespace.
    """

    def compute_metrics(p: EvalPrediction) -> Dict[str, Real]:
        # Convert model output probs/logits to predicted token IDs:
        predicted_token_ids = np.argmax(p.predictions[0], axis=2)
        # Replace everything from the first <end-of-sentence> token onward with padding (as eos
        # would terminate generation in a normal generate() call)
        for ix_batch, seq in enumerate(predicted_token_ids):
            eos_token_matches = np.where(seq == tokenizer.eos_token_id)
            if len(eos_token_matches) and len(eos_token_matches[0]):
                first_eos_posn = eos_token_matches[0][0]
                predicted_token_ids[ix_batch, first_eos_posn:] = tokenizer.pad_token_id

        gen_texts = [
            s.strip().lower()
            for s in tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)
        ]

        target_texts = [
            s.strip().lower()
            for s in tokenizer.batch_decode(
                # Replace label '-100' tokens (ignore index for BinaryCrossEntropy) with '0' (<pad>
                # token), to avoid an OverflowError when trying to decode the target text:
                np.maximum(0, p.label_ids),
                skip_special_tokens=True,
            )
        ]

        n_examples = len(gen_texts)
        n_correct = sum(1 for gen, target in zip(gen_texts, target_texts) if gen == target)
        return {
            "n_examples": len(gen_texts),
            "acc": n_correct / n_examples,
        }

    return compute_metrics
