# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#### Container image with document/image processing (and optionally OCR) tools added.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Common/base doc & image processing tools:
RUN conda install -c conda-forge poppler -y \
  && pip install amazon-textract-response-parser pdf2image "Pillow>=8,<9"

# Optional OCR engine: Tesseract+PyTesseract
# conda tesseract already includes Leptonica dependency and multi-language tessdata files by default
# (but didn't set the required TESSDATA_PREFIX variable at time of writing)
ARG INCLUDE_OCR_TESSERACT
RUN if test -z "$INCLUDE_OCR_TESSERACT" ; \
    then \
        echo Skipping OCR engine Tesseract \
    ; else \
        conda install -y -c conda-forge tesseract && \
        pip install pytesseract && \
        export TESSDATA_PREFIX='/opt/conda/share/tessdata' \
    ; fi
