#!/bin/python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Alternative train launcher script for SageMaker Training Compiler

More info at: https://docs.aws.amazon.com/sagemaker/latest/dg/training-compiler-enable.html

To use SMTC, you'll need to specify a `compiler_config` and set the "GPU_NUM_DEVICES" environment
variable on your Estimator to the number of GPUs per instance for the type you have selected. For
example:

```python
from sagemaker.huggingface import TrainingCompilerConfig

pre_estimator = HuggingFaceEstimator(
    ...,
    compiler_config=TrainingCompilerConfig(),
    env={
        ...,
        "GPU_NUM_DEVICES": "4",  # for ml.p3.8xlarge
    },
)
```

For single-GPU training, you can use the train.py entry_point as usual. However for multi-GPU
training, you'll need to instead set this `entry_point="smtc_launcher.py"` and add an additional
hyperparameter `"training_script": "train.py"`.

This training script has been tested to *functionally* work with SMTC (on Hugging Face v4.11 DLC),
but whether you'll see a useful speed-up may be quite hyperparameter- and use-case-dependent. Note
that a substantial portion of the optimization opportunity with SMTC comes from memory efficiency
allowing larger batch sizes.

Remember also that on p3.16xl and larger where it's supported, enabling SageMaker Distributed Data
Parallel can provide a useful speed boost. When *neither* SMTC nor SMDistributed are enabled, the
HF Trainer API will use PyTorch DataParallel by default (rather than DistributedDataParallel) which
can limit scaling to many GPUs - partly because memory consumption is higher on the "lead" GPU and
so CUDAOutOfMemory will be encountered at lower maximum batch sizes.

Notes from pre-training experiments
-----------------------------------

2,500 document training set (set N_DOCS_KEPT = 2500 in notebook 1) on `ml.p3.8xlarge`, pre-training
with:

- num_train_epochs = 25
- early_stopping_patience = 10
- per_device_eval_batch_size = per_device_train_batch_size
- seed = 42
- warmup_steps = 200

| SMTC | per_device_train_batch_size | learning_rate |      Execution Time  | min val loss |
|:----:|----------------------------:|--------------:|---------------------:|-------------:|
|   No |                          4  |        5e-05  | 5h28m16s (25 epochs) |    0.149301  |
|   No |                          8  |        2e-05  | 4h13m46s (25 epochs) |    0.154481  |
|  Yes |                         20  |        2e-05  |       N/A (GPU OOM)  |   N/A (OOM)  |
|  Yes |                         16  |        1e-04  | 5h03m03s (25 epochs) |    0.147910  |
|  Yes |                         16  |        5e-05  | 5h05m03s (25 epochs) |    0.141911  |
|  Yes |                         16  |        2e-05  | 5h02m52s (25 epochs) |    0.159771  |
|  Yes |                         16  |        1e-05  | 5h01m09s (25 epochs) |    0.191195  |
|  Yes |                         16  |        5e-06  | 5h01m48s (25 epochs) |    0.249820  |
|  Yes |                         12  |        1e-05  | 5h10m35s (25 epochs) |    0.165622  |
|  Yes |                          8  |        2e-05  | 2h50m02s (12 epochs) |  * 0.301963  |
|  Yes |                          8  |        1e-05  | 2h37m52s (11 epochs) |  * 0.627447  |

(*): Training unstable and stopped early after reaching `nan` loss. Best epoch reported.
"""
# Python Built-Ins:
import subprocess
import sys

if __name__ == "__main__":
    arguments_command = " ".join([arg for arg in sys.argv[1:]])
    subprocess.check_call(
        "python -m torch_xla.distributed.sm_dist " + arguments_command, shell=True
    )
