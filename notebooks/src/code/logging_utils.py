# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Logging utilities for the SageMaker code package

Provides a centralized place for re-configuring existing loggers after config load - which allows
our files to still getLogger() on import, rather than having to pass dynamic Logger objects around
everywhere between functions at call time.
"""
# Python Built-Ins:
import logging
from typing import Union

# External Dependencies:
from transformers.utils import logging as transformers_logging


transformers_logging.enable_default_handler()
transformers_logging.enable_explicit_format()

LEVEL = logging.root.level


def _create_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(LEVEL)
    return logger


LOGGER_MAP = {}


def getLogger(name: str) -> logging.Logger:
    """Retrieve or create a Logger by name"""
    if name not in LOGGER_MAP:
        LOGGER_MAP[name] = _create_logger(name)

    return LOGGER_MAP[name]


def setLevel(level: Union[int, str]):
    """Set the level of all active loggers created through this util (and HF's loggerrs)"""
    LEVEL = level
    transformers_logging.set_verbosity(LEVEL)
    for name in LOGGER_MAP:
        LOGGER_MAP[name].setLevel(LEVEL)
