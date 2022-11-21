# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Container definition for Layout+language model training & inference on SageMaker

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Core dependencies:
# - Pin PyTorch to prevent pip accidentally re-installing/upgrading it via detectron
# - Pin setuptools per https://github.com/pytorch/pytorch/issues/69894#issuecomment-1080635462
RUN pip install "amazon-textract-response-parser>=0.1,<0.2" "Pillow>=8,<9" \
    && PT_VER=`pip show torch | grep 'Version:' | sed 's/Version: //'` \
    && pip install git+https://github.com/facebookresearch/detectron2.git setuptools==59.5.0 \
        torch==$PT_VER

# Could also consider installing detectron2 via pre-built Linux wheel, depending on the PyTorch and
# CUDA versions of your base container:
# https://github.com/aws/deep-learning-containers/tree/master/huggingface/pytorch
# https://detectron2.readthedocs.io/en/latest/tutorials/install.html
#
# For example:
#     && pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html


# Additional dependencies:
# - pytesseract shouldn't be necessary after Transformers v4.18 (because we don't use Tesseract
#   OCR), but older versions have a bug: https://github.com/huggingface/transformers/issues/16845
# - datasets 1.18 and torchvision 0.11 are installed in the HF training container but missing from
#   the inference container, and we need them for inference. Upgraded datasets to use some new
#   logging controls and debug multi-worker .map() pre-processing:
RUN PT_VER=`pip show torch | grep 'Version:' | sed 's/Version: //'` \
    && pip install pytesseract "datasets==2.2.1" "torchvision>=0.11.3,<0.12" torch==$PT_VER


# If you'd like to enable this container as a Custom Image for notebook kernels, for debugging in
# SageMaker Studio, you can uncomment the below RUN command and refer to the docs for how to enable
# the image in Studio:
#
#RUN pip install ipykernel ipywidgets && python -m ipykernel install --sys-prefix
#
# https://docs.aws.amazon.com/sagemaker/latest/dg/studio-byoi.html
# https://github.com/aws-samples/sagemaker-studio-custom-image-samples
#
# An image config something like the following should work:
# {
#     "KernelSpecs": [
#         {
#             "Name": "python3",
#             "DisplayName": "Textract Transformers"
#         },
#     ],
#     "FileSystemConfig": {
#         "MountPath": "/root/data",
#         "DefaultUid": 0,
#         "DefaultGid": 0
#     }
# }
