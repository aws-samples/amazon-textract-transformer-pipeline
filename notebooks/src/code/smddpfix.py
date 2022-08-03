# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Patched HF Trainer to enable using ddp_find_unused_parameters with SageMaker Distributed
"""
# Python Built-Ins:
from logging import getLogger
from unittest.mock import patch, MagicMock

# External Dependencies:
from transformers.trainer import Trainer as TrainerBase

try:
    # v4.18+
    from transformers.utils import is_sagemaker_dp_enabled
except ImportError:
    # v4.17
    from transformers.file_utils import is_sagemaker_dp_enabled
from torch.nn.parallel import DistributedDataParallel as PTDDP
from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as SMDDP


logger = getLogger("smddpfix")


class Trainer(TrainerBase):
    """transformers.Trainer with a fix to enable ddp_find_unused_parameters on SageMaker DDP

    In at least versions 4.17.0 to 4.19.2 (probably others), HF transformers.Trainer ignores the
    ddp_find_unused_parameters argument when training with SageMaker Distributed Data Parallel.

    This customized class tries to detect and correct that behavior.
    """

    def _wrap_model(self, model, **kwargs):
        """Modified _wrap_model implementation with SageMaker ddp_find_unused_parameters fix"""
        # If the conditions for the problem don't apply, just call the original method:
        if not (is_sagemaker_dp_enabled() and self.args.ddp_find_unused_parameters):
            return super()._wrap_model(model, **kwargs)

        # In v4.18+, Trainer uses nn.parallel.DistributedDataParallel() (SM DDP is configured as a
        # backend for the vanilla PyTorch class):
        with patch(
            "transformers.trainer.nn.parallel.DistributedDataParallel",
            create=True,
        ) as ptmock:
            # In v4.17, Trainer instantiates "DDP" (as per our SMDDP above)
            with patch("transformers.trainer.DDP", create=True) as smmock:
                # (The mock/patching approach assumes that nothing in the parent function actually
                # uses the model after creating it, but that's true in the checked HF versions)
                model = super()._wrap_model(model, **kwargs)

                if len(ptmock.call_args_list):
                    # Native PyTorch DDP mock was called:
                    if len(ptmock.call_args_list) > 1:
                        raise ValueError(
                            "Error in custom fix for SageMaker Distributed Data Parallel: Native "
                            f"PyTorch DDP mock called multiple times. {ptmock.call_args_list}"
                        )
                    params = ptmock.call_args_list[0]
                    logger.info(
                        "Intercepting PyTorch DistributedDataParallel call to add "
                        "find_unused_parameters=%s",
                        self.args.ddp_find_unused_parameters,
                    )
                    params.kwargs["find_unused_parameters"] = self.args.ddp_find_unused_parameters
                    model = PTDDP(*params.args, **params.kwargs)

                elif len(smmock.call_args_list):
                    # SageMaker DDP mock was called:
                    if len(smmock.call_args_list) > 1:
                        raise ValueError(
                            "Error in custom fix for SageMaker Distributed Data Parallel: "
                            f"SageMaker DDP mock called multiple times. {smmock.call_args_list}"
                        )
                    params = smmock.call_args_list[0]
                    logger.info(
                        "Intercepting SageMaker DistributedDataParallel call to add "
                        "find_unused_parameters=%s",
                        self.args.ddp_find_unused_parameters,
                    )
                    params.kwargs["find_unused_parameters"] = self.args.ddp_find_unused_parameters
                    model = SMDDP(*params.args, **params.kwargs)

                # If model is still a mock after the updates, something's gone wrong:
                if isinstance(model, MagicMock):
                    raise ValueError(
                        "Error in custom fix for SageMaker Distributed Data Parallel: "
                        "DDP model is still mocked after checking expected cases."
                    )
        return model
