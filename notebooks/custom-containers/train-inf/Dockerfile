# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Container definition for Layout+language model training & inference on SageMaker

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Core dependencies:
# - Pin PyTorch to prevent pip accidentally re-installing/upgrading it via detectron
# - Pin setuptools per https://github.com/pytorch/pytorch/issues/69894#issuecomment-1080635462
# - Pin protobuf < 3.21 due to an error like https://stackoverflow.com/q/72441758 as of 2023-02
#   (which seems to originate from somewhere in SM DDP package when unconstrained install results
#   in downloading protobuf@4.x)
RUN pip install "amazon-textract-response-parser>=0.1,<0.2" "Pillow>=8,<9" \
    && PT_VER=`pip show torch | grep 'Version:' | sed 's/Version: //'` \
    && pip install git+https://github.com/facebookresearch/detectron2.git setuptools==59.5.0 \
        torch==$PT_VER "torchvision>=0.11.3,<0.15" "datasets>=2.4,<3" "protobuf<3.21" \
        "transformers>=4.25,<4.27"

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
    && pip install pytesseract torch==$PT_VER


# If you'd like to enable this container as a Custom Image for notebook kernels, for debugging in
# SageMaker Studio, build it with INCLUDE_NOTEBOOK_KERNEL=1 arg to include IPython kernel and also
# some other PDF processing + OCR utilities:
ARG INCLUDE_NOTEBOOK_KERNEL
RUN if test -z "$INCLUDE_NOTEBOOK_KERNEL" ; \
    then \
        echo Skipping notebook kernel dependencies \
    ; else \
        conda install -y -c conda-forge poppler tesseract && \
        PT_VER=`pip show torch | grep 'Version:' | sed 's/Version: //'` && \
        pip install easyocr ipykernel "ipywidgets>=7,<8" pdf2image pytesseract sagemaker \
            torch==$PT_VER && \
        export TESSDATA_PREFIX='/opt/conda/share/tessdata' && \
        python -m ipykernel install --sys-prefix \
    ; fi

# We would like to disable SMDEBUG when running as a notebook kernel, because it can cause some
# unwanted side-effects... But at the time of writing Dockerfile doesn't have full support for a
# conditional env statement - so:
# if --build-arg INCLUDE_NOTEBOOK_KERNEL=1, set USE_SMDEBUG to 'false', else set null.
ENV USE_SMDEBUG=${INCLUDE_NOTEBOOK_KERNEL:+false}
# ...But '' will cause problems in SM Training, default empty value to 'true' instead (which should
# be the default per:
# https://github.com/awslabs/sagemaker-debugger/blob/56fabe531692403e77ce9b5879d55211adec238e/smdebug/core/config_validator.py#L21
ENV USE_SMDEBUG=${USE_SMDEBUG:-true}

# See below guidance for adding an image built with INCLUDE_NOTEBOOK_KERNEL to SMStudio:
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
