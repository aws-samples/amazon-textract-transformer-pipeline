# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""SageMaker configuration parsing for Amazon Textract LayoutLM
"""

# Python Built-Ins:
from dataclasses import dataclass, field
import logging
import os
from typing import Optional

# External Dependencies:
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
from transformers.trainer_utils import IntervalStrategy, is_main_process


@dataclass
class SageMakerTrainingArguments(TrainingArguments):
    """Overrides & extensions to HF's CLI TrainingArguments for training LayoutLM on SageMaker

    Refer to transformers.TrainingArguments for other/base supported CLI arguments.
    """

    dataloader_num_workers: int = field(
        # A better default for single-instance, single-device, CPU-bottlenecked training:
        default=max(0, int(os.environ.get("SM_NUM_CPUS", 0)) - 2),
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (PyTorch only). "
                "0 means that the data will be loaded in the main process."
            ),
        },
    )
    disable_tqdm: Optional[bool] = field(
        # Log streams can't render progress bars like a GUI terminal
        default=True,
        metadata={"help": "TQDM progress bars are disabled by default for SageMaker/CloudWatch."},
    )
    do_eval: bool = field(
        # Users should not set this typical param directly
        default=True,
        metadata={
            "help": "This value is overridden by presence or absence of the `validation` channel"
        },
    )
    do_train: bool = field(
        # Users should not set this typical param directly
        default=True,
        metadata={"help": "This value is overridden by presence or absence of the `train` channel"},
    )
    early_stopping_patience: Optional[int] = field(
        # Add ability to control early stopping through SM CLI/hyperparameters
        default=None,
        metadata={
            "help": (
                "Stop training when the model's `metric_for_best_model` metric worsens for X "
                "evaluations (epochs by default)"
            ),
        },
    )
    early_stopping_threshold: Optional[float] = field(
        # Add ability to control early stopping through SM CLI/hyperparameters
        default=None,
        metadata={
            "help": (
                "Denote the absolute value the model's `metric_for_best_model` must improve by to "
                "avoid early stopping conditions"
            ),
        },
    )
    evaluation_strategy: IntervalStrategy = field(
        # We'd like some eval metrics by default, rather than the usual "no" strategy
        default="epoch",
        metadata={"help": "The evaluation strategy to use."},
    )
    model_dir: Optional[str] = field(
        # Add this param to differentiate checkpoint output (output_dir) from final model output
        # (model_dir).
        default="/opt/ml/model",
        metadata={
            "help": (
                "(You shouldn't need to override this on SageMaker): "
                "The output directory where the final model will be written."
            ),
        },
    )
    output_dir: Optional[str] = field(
        default=(
            "/opt/ml/checkpoints"
            if os.path.isdir("/opt/ml/checkpoints")
            else "/tmp/transformers/checkpoints"
        ),
        metadata={
            "help": (
                "(You shouldn't need to override this on SageMaker): "
                "The output directory where model checkpoints and trainer state will be written. "
                "Note HF local checkpointing defaults ON even if SageMaker S3 upload defaults OFF."
            ),
        },
    )
    overwrite_output_dir: bool = field(
        default=True,
        metadata={
            "help": (
                "Overwrite the content of the output directory."
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    # Tweak default batch sizes for this model & task
    per_device_train_batch_size: int = field(
        default=4, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )

    # Patch to align container version (v4.6.1) to more current 4.8:
    def get_process_log_level(self):
        return logging.INFO if is_main_process(self.local_rank) else logging.WARN


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to train."""

    cache_dir: Optional[str] = field(
        # Map this folder to the persistent checkpoints dir if available, or else at least pick
        # somewhere that's under the EBS mount (not the fixed-size root volume).
        default=(
            "/opt/ml/checkpoints/cache"
            if os.path.isdir("/opt/ml/checkpoints")
            else "/tmp/transformers/cache"
        ),
        metadata={
            "help": (
                "(You shouldn't need to override this on SageMaker): "
                "Caches to /opt/ml/checkpoints/cache if SaageMaker checkpointing is enabled, "
                "otherwise to a folder in the container's /tmp."
            ),
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (branch name, tag name or commit id).",
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` "
                "(necessary to use this script with private models)."
            ),
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizers (backed by the tokenizers library)"
        },
    )


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""

    annotation_attr: str = field(
        default="labels",
        metadata={"help": "Attribute name of the annotations in the manifest file"},
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training "
            "examples to this value if set."
        },
    )
    num_labels: Optional[int] = field(
        default=2,
        metadata={"help": "Number of entity classes (including 'other/none')"},
    )
    task_name: Optional[str] = field(
        default="ner",
        metadata={"help": "The name of the task (ner, mlm...)."},
    )
    textract: Optional[str] = field(
        default=os.environ.get("SM_CHANNEL_TEXTRACT"),
        metadata={"help": "The data channel containing Textract JSON results"},
    )
    textract_prefix: str = field(
        default="",
        metadata={"help": "Prefix mapping manifest S3 URIs to the 'textract' channel"},
    )
    train: Optional[str] = field(
        default=os.environ.get("SM_CHANNEL_TRAIN"),
        metadata={"help": "The data channel (local folder) for training"},
    )
    validation: Optional[str] = field(
        default=os.environ.get("SM_CHANNEL_VALIDATION"),
        metadata={"help": "The data channel (local folder) for model evaluation"},
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to return all the entity-level scores during evaluation or just the "
                "overall metrics."
            ),
        },
    )

    def __post_init__(self):
        if not (self.train and self.textract):
            raise ValueError(
                "'train' (JSONLines manifest file) and 'textract' (Folder of Textract result "
                "JSONs) channels are mandatory"
            )
        self.task_name = self.task_name.lower()


def parse_args(cmd_args=None):
    """Parse config arguments from the command line, or cmd_args instead if supplied"""

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SageMakerTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=cmd_args)

    # Auto-set activities depending which channels were provided:
    training_args.do_train = bool(data_args.train)
    training_args.do_eval = bool(data_args.validation)

    return model_args, data_args, training_args
