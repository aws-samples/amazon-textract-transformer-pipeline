# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Container image with document/image processing tools added.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN conda install -c conda-forge poppler -y \
  && pip install amazon-textract-response-parser pdf2image "Pillow>=8,<9"
