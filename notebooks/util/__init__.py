# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Utility functions to help keep the notebooks tidy - Amazon Textract + Transformers sample
"""

# Before importing any submodules, we'll configure Python `logging` nicely for notebooks.
#
# By "nicely", we mean:
#  - Setting up the formatting to display timestamp, logger name and level
#  - Sending messages >= WARN to stderr (so Jupyter renders them with pink/red background)
#  - Sending messages < WARN to stdout (so Jupyter renders them plain, like a print())
import logging
from logging.config import dictConfig


class MaxLevelFilter(logging.Filter):
    """A custom Python logging Filter to reject messages *above* a certain `max_level`"""

    def __init__(self, max_level):
        self._max_level = max_level
        super().__init__()

    def filter(self, record):
        return record.levelno <= self._max_level

    @classmethod
    def qualname(cls):
        return ".".join([cls.__module__, cls.__qualname__])


dictConfig(
    {
        "formatters": {
            "basefmt": {"format": "%(asctime)s %(name)s [%(levelname)s] %(message)s"},
        },
        "filters": {
            "maxinfo": {"()": MaxLevelFilter.qualname(), "max_level": logging.INFO},
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "filters": ["maxinfo"],
                "formatter": "basefmt",
                "level": logging.DEBUG,
                "stream": "ext://sys.stdout",
            },
            "stderr": {
                "class": "logging.StreamHandler",
                "formatter": "basefmt",
                "level": logging.WARN,
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {
            "": {"handlers": ["stderr", "stdout"], "level": logging.INFO},
        },
        "version": 1,
    }
)

# Now import the submodules:
from . import deployment
from . import ocr
from . import preproc
from . import project
from . import s3
from . import smgt
from . import training
from . import uid
from . import viz
