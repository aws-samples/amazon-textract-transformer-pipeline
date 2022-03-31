# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Train HuggingFace LayoutLM on Amazon Textract results
"""
# Python Built-Ins:
import os
import shutil

# External Dependencies:
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    PreTrainedTokenizerFast,
    set_seed,
    Trainer,
)
from transformers.trainer_utils import get_last_checkpoint

# Local Dependencies:
from . import config
from . import data
from . import logging_utils


logger = logging_utils.getLogger("main")


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

    if data_args.task_name == "ner":
        model = AutoModelForTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif data_args.task_name == "mlm":
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        raise ValueError(
            f"Unknown data_args.task_name '{data_args.task_name}' not in ('mlm', 'ner')"
        )
    return config, model, tokenizer


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
    datasets = data.get_datasets(data_args, tokenizer)
    if datasets.train_dataset:
        logger.info(f"train dataset has {len(datasets.train_dataset)} samples")
    else:
        logger.info("No training dataset provided")
    if datasets.eval_dataset:
        logger.info(f"validation dataset has {len(datasets.eval_dataset)} samples")
    else:
        logger.info("No validation dataset provided")

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

    logger.info("Setting up trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets.train_dataset,
        eval_dataset=datasets.eval_dataset if data_args.validation else None,
        tokenizer=tokenizer,
        data_collator=datasets.data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=training_args.early_stopping_patience,
                early_stopping_threshold=training_args.early_stopping_threshold,
            )
        ]
        if (
            training_args.early_stopping_patience is not None
            or training_args.early_stopping_threshold is not None
        )
        else [],
        compute_metrics=datasets.metric_computer,
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
            else len(datasets.train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(datasets.train_dataset))

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
    for currpath, _, files in os.walk("."):
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


def main():
    """CLI script entry point to parse arguments and run training"""
    model_args, data_args, training_args = config.parse_args()

    # Logging setup:
    log_level = training_args.get_process_log_level()
    logging_utils.setLevel(log_level)

    logger.info("Loaded arguments:\n%s\n%s\n%s", model_args, data_args, training_args)
    logger.info("Starting!")
    set_seed(training_args.seed)

    # Start training:
    train(model_args, data_args, training_args)
