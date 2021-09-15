# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Train HuggingFace LayoutLM on Amazon Textract results"""

# Python Built-Ins:
import logging
import os
import shutil
import sys

# Initial logging setup before other imports (level to be overridden by CLI later):
if __name__ == "__main__":
    consolehandler = logging.StreamHandler(sys.stdout)
    consolehandler.setFormatter(
        logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s")
    )
    logging.basicConfig(handlers=[consolehandler], level=os.environ.get("LOG_LEVEL", logging.INFO))

# External Dependencies:
import numpy as np
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    PreTrainedTokenizerFast,
    set_seed,
    Trainer,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging as transformers_logging

# Local Dependencies:
import config
import data

# Import everything defined by inference.py to enable directly deploying this model via SageMaker
# SDK's Estimator.deploy(), which will leave env var SAGEMAKER_PROGRAM=train.py:
from inference import *


logger = logging.getLogger()


def get_model(model_args, data_args):
    """Load pre-trained Config, Model and Tokenizer"""
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=data_args.num_labels,
        label2id={str(i): i for i in range(data_args.num_labels)},
        id2label={i: str(i) for i in range(data_args.num_labels)},
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        # Potentially unnecessary extra kwargs for LayoutLM:
        max_position_embeddings=data_args.max_seq_length,  # TODO: VALIDATE THIS
        max_2d_position_embeddings=2 * data_args.max_seq_length,
    )

    tokenizer_name_or_path = (
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    )
    if config.model_type in {"gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            add_prefix_space=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    return config, model, tokenizer


def get_metric_computer(num_labels, pad_token_label_id):
    """Generate the compute_metrics callable for a HF Trainer

    Note that we don't use seqeval (like some standard TokenClassification examples) because our
    tokens are not guaranteed to be in reading order by the nature of LayoutLM/OCR: So conventional
    Inside-Outside-Beginning (IOB/ES) notation doesn't really make sense.
    """
    other_class_label = num_labels - 1

    def compute_metrics(p):
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


def train(model_args, data_args, training_args):
    logger.info("Creating config and model")
    _, model, tokenizer = get_model(model_args, data_args)

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. See the list "
            "at https://huggingface.co/transformers/index.html#supported-frameworks for details."
        )

    logger.info("Loading datasets")
    pad_token_label_id = CrossEntropyLoss().ignore_index
    train_dataset = data.get_dataset(data_args.train, tokenizer, data_args, pad_token_label_id)
    logger.info(f"train dataset has {len(train_dataset)} samples")
    if data_args.validation:
        val_dataset = data.get_dataset(
            data_args.validation,
            tokenizer,
            data_args,
            pad_token_label_id,
        )
        logger.info(f"validation dataset has {len(val_dataset)} samples")
    else:
        val_dataset = None

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            logger.warning("No previous checkpoint found: training from scratch")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this "
                "behavior, create the training job with an empty `checkpoint_s3_uri` or none."
            )

    compute_metrics = get_metric_computer(data_args.num_labels, pad_token_label_id)

    logger.info("Setting up trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if data_args.validation else None,
        tokenizer=tokenizer,
        data_collator=data.DummyDataCollator(),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=(
                    1
                    if training_args.early_stopping_patience is None
                    else training_args.early_stopping_patience
                ),
                early_stopping_threshold=training_args.early_stopping_threshold or 0.0,
            )
        ]
        if (
            training_args.early_stopping_patience is not None
            or training_args.early_stopping_threshold is not None
        )
        else [],
        compute_metrics=compute_metrics,
    )

    if not training_args.do_train:
        logger.warning(f"Training skipped (args.do_train={training_args.do_train})")
    else:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # (Saves the tokenizer too)

        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    logger.info(f"Saving model to {training_args.model_dir}")
    trainer.save_model(training_args.model_dir)  # (Saves the tokenizer too)

    # To enable directly deploying this model via SageMaker SDK's Estimator.deploy() (rather than
    # needing to create a PyTorchModel with entry_point / source_dir args), we need to save any
    # inference handler function code to model_dir/code. Here we compromise efficiency to the
    # benefit of usage simplicity, by just copying the contents of this training code folder to the
    # model/code folder for inference:
    code_path = os.path.join(training_args.model_dir, "code")
    logger.info(f"Copying code to {code_path} for inference")
    for currpath, dirs, files in os.walk("."):
        for file in files:
            # Skip any filenames starting with dot:
            if file.startswith("."):
                continue
            filepath = os.path.join(currpath, file)
            # Skip any pycache or dot folders:
            if ((os.path.sep + ".") in filepath) or ("__pycache__" in filepath):
                continue
            relpath = filepath[len(".") :]
            if relpath.startswith(os.path.sep):
                relpath = relpath[1:]
            outpath = os.path.join(code_path, relpath)
            logger.info(f"Copying {filepath} to {outpath}")
            os.makedirs(outpath.rpartition(os.path.sep)[0], exist_ok=True)
            shutil.copy2(filepath, outpath)

    return trainer


# Since we want to support targeting this train.py as a valid import entry point for inference too,
# we need to only run the actual training routine if run as a script, not when imported as module:
if __name__ == "__main__":
    model_args, data_args, training_args = config.parse_args()

    # Logging setup:
    log_level = training_args.get_process_log_level()
    transformers_logging.set_verbosity(log_level)
    transformers_logging.enable_default_handler()
    transformers_logging.enable_explicit_format()
    for l in (logger, data.logger):
        logger.setLevel(log_level)

    logger.info("Loaded arguments:\n%s\n%s\n%s", model_args, data_args, training_args)
    logger.info("Starting!")
    set_seed(training_args.seed)

    # Start training:
    train(model_args, data_args, training_args)
