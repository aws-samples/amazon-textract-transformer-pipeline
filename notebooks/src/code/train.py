# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Train HuggingFace LayoutLM on Amazon Textract results

(This script also allows for training non-layout-aware models e.g. T5 on seq2seq conditional text
generation task)
"""
# Python Built-Ins:
from inspect import signature
import os
import shutil
from typing import Optional, Tuple

# External Dependencies:
from torch import distributed as dist
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoProcessor,
    AutoTokenizer,
    EarlyStoppingCallback,
    LayoutLMv2Config,
    LayoutXLMProcessor,
    LayoutXLMTokenizerFast,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    ProcessorMixin,
    set_seed,
    Trainer,
)
from transformers.file_utils import EntryNotFoundError
from transformers.trainer_utils import get_last_checkpoint

# Local Dependencies:
from . import config
from . import data
from . import logging_utils
from .smddpfix import Trainer
from .models.layoutlmv2 import LayoutLMv2ForPretraining

logger = logging_utils.getLogger("main")


def get_model(
    model_args: config.ModelArguments, data_args: config.DataTrainingArguments
) -> Tuple[PretrainedConfig, PreTrainedModel, PreTrainedTokenizerFast, Optional[ProcessorMixin]]:
    """Load pre-trained Config, Model, Tokenizer, and Processor if one exists"""
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=data_args.num_labels,
        label2id={str(i): i for i in range(data_args.num_labels)},
        id2label={i: str(i) for i in range(data_args.num_labels)},
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        # For LayoutLMV1 we used to explicitly set:
        # max_position_embeddings=data_args.max_seq_length,
        # max_2d_position_embeddings=2 * data_args.max_seq_length,
        # ...But the LayoutLMV2 tokenizer has max_position_embeddings=514 (+2) and 2d=1024... so
        # rather handle the inconsistency, we'll ignore it as basic usage won't need to set them.
    )

    tokenizer_name_or_path = (
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    )

    try:
        # AutoTokenizer doesn't detect XLM and instantiates a LayoutLMv2Processor instead!
        ProcessorClass = (
            LayoutXLMProcessor if "xlm" in tokenizer_name_or_path.lower() else AutoProcessor
        )
        processor = ProcessorClass.from_pretrained(
            model_args.model_name_or_path,
            # Feature Extractor overrides:
            apply_ocr=False,  # We use Amazon Textract
            do_resize=False,  # External thumbnailing service handles this
            # Tokenizer overrides:
            only_label_first_subword=False,  # We aggregate over all word token labels
            use_fast=True,
            # Download settings:
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        if hasattr(processor, "tokenizer"):
            tokenizer = processor.tokenizer
        elif isinstance(processor, PreTrainedTokenizerBase):
            # AutoProcessor loaded something, but it's just a standard tokenizer.
            # This happens e.g. with t5-base model as at HF transformers==4.25.1
            tokenizer = processor
            processor = None
        else:
            tokenizer = None
    except (EntryNotFoundError, OSError):
        processor = None
        tokenizer = None
    except ValueError as ve:
        if "unrecognized processor" in str(ve).lower():
            processor = None
            tokenizer = None
        else:
            raise ve
    if not processor:
        logger.info(
            "This model type does not have a Processor: %s",
            model_args.model_name_or_path,
        )

    if not tokenizer:
        if "xlm" in tokenizer_name_or_path.lower():
            # AutoTokenizer doesn't detect XLM and instantiates a LayoutLMv2Tokenizer instead!
            tokenizer = LayoutXLMTokenizerFast.from_pretrained(
                tokenizer_name_or_path,
                only_label_first_subword=False,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        elif config.model_type in {"gpt2", "roberta"}:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path,
                only_label_first_subword=False,
                cache_dir=model_args.cache_dir,
                use_fast=True,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                add_prefix_space=True,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path,
                only_label_first_subword=False,
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
        if isinstance(config, LayoutLMv2Config):
            logger.info(
                "As of v4.18, HF transformers does not bundle a variant of LayoutLMv2/XLM for "
                "pre-training. Using a custom implementation which may not exactly align to the "
                "published research."
            )
            model = LayoutLMv2ForPretraining.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            model = AutoModelForMaskedLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    elif data_args.task_name == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(
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
    return config, model, tokenizer, processor


def train(
    model_args: config.ModelArguments,
    data_args: config.DataTrainingArguments,
    training_args: config.SageMakerTrainingArguments,
) -> Trainer:
    training_args._setup_devices  # Force distributed setup if applicable and not already done
    logger.info("Started with local_rank %s", training_args.local_rank)
    # Don't strictly need this around the model setup too, but keeps logs more understandable:
    # Using the HF decorator rather than torch.distributed.barrier() to try and keep a bit more
    # environment-agnostic:
    with training_args.main_process_first(desc="Waiting for main process to load model and data"):
        logger.info("Creating config and model")
        _, model, tokenizer, processor = get_model(model_args, data_args)

        if hasattr(model, "layoutlmv2") and training_args.n_gpu > 1:
            if dist.is_initialized():
                logger.info("Synchronizing LayoutLMv2 visual batch norm for distributed training")
                model.layoutlmv2.visual.synchronize_batch_norm()
            else:
                raise ValueError(
                    "For multi-GPU training, LayoutLMv2/XLM must be run in Distributed Data "
                    "Parallel mode (PyTorch native or SageMaker Distributed). Consider using SM "
                    "DDP on a supported instance type (e.g. ml.p3.16xlarge), OR launching native "
                    "via PyTorch DDP via ddp_launcher.py entrypoint"
                )
                # For more information, see:
                # https://github.com/NielsRogge/Transformers-Tutorials/issues/30
                # https://github.com/huggingface/transformers/issues/14110
                # https://sagemaker.readthedocs.io/en/stable/api/training/smd_data_parallel_use_sm_pysdk.html
                # For SM Distributed, ddp_launcher.py is not necessary - point straight to train.py

        # Tokenizer check: Our MLM/NER data prep requires a fast tokenizer.
        if data_args.task_name in ("mlm", "ner") and not isinstance(
            tokenizer, PreTrainedTokenizerFast
        ):
            raise ValueError(
                "This example script only works for models that have a fast tokenizer. See the list "
                "at https://huggingface.co/transformers/index.html#supported-frameworks for details."
            )

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

        # Was going to close the old main_process_first context and start a separate one here to
        # maximize the time available for dataset prep until the default DDP 30min timeout kicks
        # in... But doing so seems to cause datasets to deadlock/freeze after splitting pages for
        # the training dataset. I think the same happens if using torch.distributed.barrier too?
        logger.info("Loading datasets")
        datasets = data.get_datasets(
            data_args,
            tokenizer,
            processor,
            model_param_names=set(signature(model).parameters),
            n_workers=training_args.dataproc_num_workers,
            cache_dir=model_args.cache_dir,
        )

        if datasets.train_dataset:
            logger.info(f"train dataset has {len(datasets.train_dataset)} samples")
        else:
            logger.info("No training dataset provided")
        if datasets.eval_dataset:
            logger.info(f"validation dataset has {len(datasets.eval_dataset)} samples")
        else:
            logger.info("No validation dataset provided")

    logger.info("Setting up trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets.train_dataset,
        eval_dataset=datasets.eval_dataset,
        # No `tokenizer`, as either the dataset or the data_collator does it for us
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
        trainer.save_model()

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
    trainer.save_model(training_args.model_dir)
    if processor:
        # (processor saves tokenizer anyway)
        processor.save_pretrained(os.path.join(training_args.model_dir))
    else:
        tokenizer.save_pretrained(os.path.join(training_args.model_dir))

    # To enable directly deploying this model via SageMaker SDK's Estimator.deploy() (rather than
    # needing to create a PyTorchModel with entry_point / source_dir args), we need to save any
    # inference handler function code to model_dir/code. Here we compromise efficiency to the
    # benefit of usage simplicity, by just copying the contents of this training code folder to the
    # model/code folder for inference:
    code_path = os.path.join(training_args.model_dir, "code")
    if not os.path.abspath(".").startswith("/opt/ml/"):
        logger.warning(
            "Skipping output code copy step: Seems not to be running inside SageMaker job"
        )
        # If you try to recursively copy '.' in, for example, a SMStudio environment where '.' is
        # the notebooks/ folder (not notebooks/src) and notebooks/data is populated - you could be
        # waiting a very... long... time... Just create an empty folder to demonstrate:
        os.makedirs(code_path, exist_ok=True)
    else:
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


def main() -> None:
    """CLI script entry point to parse arguments and run training"""
    model_args, data_args, training_args = config.parse_args()

    # Logging setup:
    log_level = training_args.get_process_log_level()
    logging_utils.setLevel(log_level)

    logger.info("Loaded arguments:\n%s\n%s\n%s", model_args, data_args, training_args)
    logger.info("Starting!")
    if training_args.seed:
        set_seed(training_args.seed)
    else:
        logger.info("Random seed not set - results will be non-deterministic")

    # Start training:
    train(model_args, data_args, training_args)
