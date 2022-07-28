# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""(Native PyTorch) DistributedDataParallel launcher

Use this entrypoint script to launch training with native PyTorch DDP on SageMaker. You don't need
it if using SageMaker DDP - in which case directly set 'train.py' as your entrypoint.
"""
# Python Built-Ins:
import json
import os
import socket
import subprocess
import sys


# Path to resource config file IF running on SageMaker:
SM_CONFIG_PATH = "/opt/ml/input/config/resourceconfig.json"

if __name__ != "__main__":
    # If the file is imported as a module, we're in inference mode and should pass through the
    # override functions defined in the inference module. This is to support directly deploying the
    # model via SageMaker SDK's Estimator.deploy(), which will carry over the environment variable
    # SAGEMAKER_PROGRAM=ddp_launcher.py from training - causing the server to try and load handlers
    # from here rather than inference.py.
    from code.inference import *
else:
    if os.path.exists(SM_CONFIG_PATH):
        # Running on SageMaker: Load distribution configs from the resourceconfig file
        with open(SM_CONFIG_PATH) as file:
            cluster_config = json.load(file)

        host_names = cluster_config["hosts"]
        default_n_nodes = len(host_names)
        default_node_rank = host_names.index(os.environ.get("SM_CURRENT_HOST"))

        # Elect first listed host as the leader for PyTorch DDP
        print("CLUSTER HOSTS:")
        host_ips = [socket.gethostbyname(host) for host in host_names]
        for ix, host in enumerate(host_names):
            print(
                " - {}host: {}, IP: {}".format(
                    "(leader) " if ix == 0 else "",
                    host,
                    host_ips[ix],
                )
            )
        leader = host_ips[0]

        # Set the network interface for inter node communication
        os.environ["NCCL_SOCKET_IFNAME"] = cluster_config["network_interface_name"]

    else:
        # Seems not to be a SageMaker training job (could be e.g. testing on notebook, local).
        # Default to single-machine setup:
        default_n_nodes = 1
        default_node_rank = 0
        leader = "127.0.0.1"

    # Set up DDP & NCCL environment variables:
    # https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html#ncclknobs
    # https://github.com/aws/sagemaker-pytorch-training-toolkit/blob/88ca48a831bf4f099d4c57f3c18e0ff92fa2b48c/src/sagemaker_pytorch_container/training.py#L103
    #
    # Disable IB transport and force to use IP sockets by default:
    os.environ["NCCL_IB_DISABLE"] = "1"
    # Set NCCL log level (could be INFO for more debugging information):
    if not os.environ.get("NCCL_DEBUG"):
        os.environ["NCCL_DEBUG"] = "WARN"

    # Launch PyTorch DDP:
    ddp_cmd = (
        [
            "python",
            "-m",
            "torch.distributed.launch",
            "--nproc_per_node",
            os.environ["SM_NUM_GPUS"],
            "--nnodes",
            str(default_n_nodes),
            "--node_rank",
            str(default_node_rank),
            "--master_addr",
            leader,
            "--master_port",
            "7777",
        ]
        # ...And pass through arguments for the actual train script:
        + ["train.py"]
        + [arg for arg in sys.argv[1:]]
    )
    print("LAUNCHING: " + " ".join(ddp_cmd))
    subprocess.check_call(ddp_cmd)
