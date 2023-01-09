# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""SageMaker configuration parsing for Amazon Textract LayoutLM
"""

# Python Built-Ins:
from dataclasses import dataclass, field
import os
import tarfile
from typing import Optional, Sequence, Tuple

# External Dependencies:
try:
    from datasets import disable_progress_bar as disable_datasets_progress_bar
except ImportError:  # Not available in datasets <v2.0.0
    disable_datasets_progress_bar = None
from torch import use_deterministic_algorithms
from transformers import HfArgumentParser, TrainingArguments
from transformers.trainer_utils import IntervalStrategy


def get_n_cpus() -> int:
    return int(os.environ.get("SM_NUM_CPUS", len(os.sched_getaffinity(0))))


def get_n_gpus() -> int:
    return int(os.environ.get("SM_NUM_GPUS", 0))


def get_default_num_workers() -> int:
    """Choose a sensible default dataloader_num_workers based on available hardware"""
    n_cpus = get_n_cpus()
    n_gpus = get_n_gpus()
    # Don't create so many workers you lock all processes into resource contention:
    max_workers = max(0, n_cpus - 2)
    if n_gpus:
        # Don't create unnecessarily high numbers of workers per GPU:
        # (Which can cause CUDAOutOfMemory e.g. on p3.16xlarge, or RAM exhaustion with SageMaker
        # Training Compiler)
        max_workers = min(
            max_workers,
            max(8, n_gpus * 4),
        )

    return max(0, max_workers)


@dataclass
class SageMakerTrainingArguments(TrainingArguments):
    """Overrides & extensions to HF's CLI TrainingArguments for training LayoutLM on SageMaker

    Refer to transformers.TrainingArguments for other/base supported CLI arguments.
    """

    dataloader_num_workers: int = field(
        # A better default for single-instance, single-device, CPU-bottlenecked training:
        default=get_default_num_workers(),
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (PyTorch only). "
                "0 means that the data will be loaded in the main process."
            ),
        },
    )
    dataproc_num_workers: Optional[int] = field(
        # Our data pre-processing is explicitly configured to run in the lead process and then load
        # from cache for other processes - so we can use lots of workers because the lead proc will
        # be running it alone:
        default=max(1, get_n_cpus() - 2),
        metadata={
            "help": (
                "Number of subprocesses to use for data preparation before training commences. "
                "0 means that the data will be loaded in the main process (Good for debug)."
            ),
        },
    )
    ddp_find_unused_parameters: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "For DistributedDataParallel training, LayoutLMv2/XLM require "
                "find_unused_parameters=True because of unused parameters in the model structure. "
                "For LLMv1 in DDP mode you could turn this off for a performance boost."
            )
        },
    )
    disable_tqdm: Optional[bool] = field(
        # Log streams can't render progress bars like a GUI terminal
        # NOTE: If you run into problems debugging dataset prep, you may like to enable progress
        default=True,
        metadata={"help": "TQDM progress bars are disabled by default for SageMaker/CloudWatch."},
    )
    do_eval: bool = field(
        default=None,
        metadata={
            "help": (
                "This value is normally set by the presence or absence of the 'validation' "
                "channel, but can be explicitly overridden."
            )
        },
    )
    do_train: bool = field(
        default=True,
        metadata={"help": "Set false to disable training (for validation-only jobs)"},
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
    full_determinism: bool = field(
        # (This will be a standard TrainingArg as of 4.19.0, but isn't in the current 4.17)
        default=False,
        metadata={"help": ("Will be automatically enabled for this script if `seed` is truthy.")},
    )
    save_strategy: IntervalStrategy = field(
        # Should match evaluation strategy for early stopping to work
        default="epoch",
        metadata={"help": "The model save strategy to use."},
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
            ),
        },
    )
    # Tweak default batch sizes for this model & task
    per_device_train_batch_size: int = field(
        default=4, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to automatically remove datasets.Dataset columns unused by the "
                "model.forward() method. This should be False by default, as our implementation "
                "either uses a custom data collator (LLMv1) or pre-processes the dataset (v2/XLM)."
            ),
        },
    )

    def __post_init__(self):
        super().__post_init__()
        # HF datasets library requires n_proc = None, not 0, if workers are disabled:
        if not self.dataproc_num_workers:
            self.dataproc_num_workers = None
        # ...And it doesn't see the TrainingArguments progress setting by default:
        if self.disable_tqdm and disable_datasets_progress_bar:
            disable_datasets_progress_bar()
        # This script uses cuBLAS operators that require a workspace to run in deterministic /
        # reproducible mode. You could alternatively set ":16:8" to save (about 24MiB of) GPU
        # memory at the cost of performance. More information at:
        # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        if self.seed:
            self.full_determinism = True
            if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
                print(
                    "Defaulting CUBLAS_WORKSPACE_CONFIG=':4096:8' to enable deterministic ops as "
                    "`seed` is set."
                )
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            use_deterministic_algorithms(True)
        # Normalize early stopping configuration if it seems enabled:
        if self.early_stopping_patience is not None or self.early_stopping_threshold is not None:
            # The EarlyStoppingCallback requires load_best_model_at_end=True:
            self.load_best_model_at_end = True
            # Also make sure the early stopping settings default sensibly if turning it on:
            if self.early_stopping_patience is None:
                self.early_stopping_patience = 1
            if not self.early_stopping_threshold:
                self.early_stopping_threshold = 0.0


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
        default=os.environ.get("SM_CHANNEL_MODEL_NAME_OR_PATH"),
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Usually, either set this as a name in SageMaker hyperparameter (e.g. "
            "'microsoft/layoutlm-base-uncased') or as an S3 URI in SageMaker channel (e.g. "
            "estimator.fit({'model_name_or_path': 's3://...tar.gz'}). Leave unset if you want to "
            "train a model from scratch."
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

    def __post_init__(self):
        # Extract pretrained model if provided as tarball / folders containing tarball:
        if os.path.isdir(self.model_name_or_path):
            contents = os.listdir(self.model_name_or_path)
            print(f"Got pretrained model folder with contents: {contents}")
            tar_candidates = list(filter(lambda f: f.lower().endswith(".tar.gz"), contents))
            n_tar_candidates = len(tar_candidates)
            if n_tar_candidates == 1:
                # (This is the path that gets used when supplying prev trained model as a channel)
                print(f"Extracting model tarball {tar_candidates[0]} to {self.model_name_or_path}")
                with tarfile.open(
                    os.path.join(self.model_name_or_path, tar_candidates[0])
                ) as tarmodel:
                    tarmodel.extractall(self.model_name_or_path)
                print(f"Model folder top-level contents: {os.listdir(self.model_name_or_path)}")
            elif n_tar_candidates > 1:
                raise ValueError("model_name_or_path data channel contains >1 .tar.gz file")
            else:
                print(
                    "No tarballs present in input model_name_or_path folder - assuming already "
                    "extracted"
                )
        elif os.path.isfile(self.model_name_or_path):
            modelfolder = os.path.dirname(self.model_name_or_path)
            print(f"Extracting model tarball {self.model_name_or_path} to {modelfolder}")
            with tarfile.open(self.model_name_or_path) as tarmodel:
                tarmodel.extractall(modelfolder)
            print(f"Model folder top-level contents: {os.listdir(modelfolder)}")


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""

    annotation_attr: str = field(
        default="labels",
        metadata={"help": "Attribute name of the annotations in the manifest file"},
    )
    dataproc_batch_size: int = field(
        default=16,
        metadata={
            # RAM usage in tests on p3/g4dn instances suggests this could probably be much higher
            # if needed, but I'm not sure whether it'd actually improve speed.
            "help": "Base batch size for (up-front, before training) data pre-processing."
        },
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be split."
        },
    )
    pad_to_multiple_of: Optional[int] = field(
        default=8,
        metadata={"help": "Pad sequences to a multiple of this value, for tensor core efficiency"},
    )
    # TODO: Check this is observed correctly
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training "
            "examples to this value if set."
        },
    )
    task_name: Optional[str] = field(
        default="ner",
        metadata={
            "help": "The name of the task. This script currently supports 'ner' for entity "
            "recognition, 'mlm' for pre-training (masked language modelling), or 'seq2seq' for "
            "experimental (non-layout-aware) sequence-to-sequence data normalizations."
        },
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
    images: Optional[str] = field(
        default=os.environ.get("SM_CHANNEL_IMAGES"),
        metadata={"help": "The data channel containing (resized) page images"},
    )
    images_prefix: str = field(
        default="",
        metadata={"help": "Prefix mapping manifest S3 URIs to the 'images' channel"},
    )

    # NER (token classification) specific parameters:
    num_labels: Optional[int] = field(
        default=2,
        metadata={"help": "Number of entity classes (including 'other/none')"},
    )
    # TODO: Implement or remove
    # return_entity_level_metrics: bool = field(
    #     default=False,
    #     metadata={
    #         "help": (
    #             "Whether to return all the entity-level scores during evaluation or just the "
    #             "overall metrics."
    #         ),
    #     },
    # )

    # MLM (pre-training) specific parameters:
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    tiam_probability: float = field(
        default=0.15,
        metadata={
            "help": "Ratio of text lines to mask in the image for LayoutLMv2/XLM 'Text-Image "
            "Alignment' pre-training loss. Set 0 to disable TIA in pre-training. Ignored for LLMv1."
        },
    )
    tim_probability: float = field(
        default=0.2,
        metadata={
            "help": "Ratio of page images to randomly permute for LayoutLMv2/XLM 'Text-Image "
            "Matching' pre-training loss. Set 0 to disable TIM in pre-training. Ignored for LLMv1.",
        },
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()
        if (not self.textract) and (self.task_name != "seq2seq"):
            raise ValueError("'textract' (Folder of Textract result JSONs) channel is mandatory")


def parse_args(
    cmd_args: Optional[Sequence[str]] = None,
) -> Tuple[ModelArguments, DataTrainingArguments, SageMakerTrainingArguments]:
    """Parse config arguments from the command line, or cmd_args instead if supplied"""

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SageMakerTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=cmd_args)

    # Auto-set activities depending which channels were provided.
    # By only overriding do_eval if it's not explicitly specified, we allow override e.g. to force
    # validation in a job where no external dataset is provided but a synthetic one can be generated
    if training_args.do_eval is None:
        training_args.do_eval = bool(data_args.validation)
    if not training_args.do_eval:
        training_args.evaluation_strategy = "no"

    return model_args, data_args, training_args
