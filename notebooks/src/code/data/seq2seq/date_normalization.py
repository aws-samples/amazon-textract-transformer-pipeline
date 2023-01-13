# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Synthetic dataset generation for date normalization via seq2seq text modelling

This script provides utilities for tackling date field format normalization as a conditional
language modelling task. For example, training a model with input like:

    "Convert dates to YYYY-MM-DD: 31/12/2000"

...into a target output sequence like "2000-12-31".

In the plain text case, it's relatively straightforward to generate synthetic data for this task
as shown here. By modifying the distribution of randomly generated dates, the likelihood of
different observed formats in the source document and target formats in the prompt, we can tailor
model performance to match target use-case without having to write extensive text parsing rules.
"""
# Python Built-Ins:
from dataclasses import dataclass
from logging import getLogger
import time
from typing import List, Optional, Sequence

# External Dependencies:
from datasets import Dataset, DatasetInfo
import numpy as np


logger = getLogger("data.seq2seq.dates")


@dataclass
class DateFormatConfig:
    """Configuration describing a date format for date normalization tasks

    Parameters
    ----------
    format_str :
        A formal `time.strftime`-compatible specifier for the date format, for example `%Y-%m-%d`.
    format_name :
        A human-friendly identifier for the format, as might be used in task prompts. For example
        `YYYY-MM-DD` for a prompt like "Convert dates to YYYY-MM-DD".
    observed_weight :
        Weight/frequency with which this date format will be observed in content, for synthetic data
        generation. Does not need to be normalized to 1.0 across all your configured formats,
        because the dataset generator will ensure this for you.
    target_weight :
        Weight/frequency with which this date format will be used as the target for prompting, for
        synthetic data generation. Does not need to be normalized to 1.0 across all your configured
        formats, because the dataset generator will ensure this for you.
    """

    format_str: str
    format_name: str
    observed_weight: float
    target_weight: float


# Format configuration for synthetic date normalization training data generation
DATE_FORMAT_CONFIGS = [
    DateFormatConfig("%Y-%m-%d", "YYYY-MM-DD", observed_weight=0.1, target_weight=0.7),
    DateFormatConfig("%m/%d/%y", "MM/DD/YY", observed_weight=0.35, target_weight=0.05),
    DateFormatConfig("%m/%d/%Y", "MM/DD/YYYY", observed_weight=0.35, target_weight=0.2),
    DateFormatConfig("%d/%m/%y", "DD/MM/YY", observed_weight=0.05, target_weight=0.02),
    DateFormatConfig("%d/%m/%Y", "DD/MM/YYYY", observed_weight=0.04, target_weight=0.03),
    # Including day names and month names:
    DateFormatConfig("%A %b %d %y", "DDDD MMM DD YY", observed_weight=0.03, target_weight=0.0),
    DateFormatConfig("%A, %b %d %y", "DDDD, MMM DD YY", observed_weight=0.02, target_weight=0.0),
    DateFormatConfig("%a %b %d, %y", "DDD MMM DD, YY", observed_weight=0.02, target_weight=0.0),
    DateFormatConfig("%a. %b %d %y", "DDD. MMM DD YY", observed_weight=0.02, target_weight=0.0),
    DateFormatConfig("%A %b %dst %y", "DDDD MMM DDst YY", observed_weight=0.01, target_weight=0.0),
    DateFormatConfig("%A %b %dnd %y", "DDDD MMM DDnd YY", observed_weight=0.01, target_weight=0.0),
    DateFormatConfig("%A %b %drd %y", "DDDD MMM DDrd YY", observed_weight=0.01, target_weight=0.0),
    DateFormatConfig("%A %b %dth %y", "DDDD MMM DDth YY", observed_weight=0.01, target_weight=0.0),
    DateFormatConfig("%a %d %b %y", "DDD DD MMM YY", observed_weight=0.02, target_weight=0.0),
    DateFormatConfig("%a. %d %b %y", "DDD. DD MMM YY", observed_weight=0.02, target_weight=0.0),
    # Including times:
    DateFormatConfig(
        "%Y-%m-%d %H:%M:%S", "YYYY-MM-DD HH:mm:ss", observed_weight=0.02, target_weight=0.0
    ),
    DateFormatConfig("%d/%m/%y %H:%M", "DD/MM/YY HH:mm", observed_weight=0.02, target_weight=0.0),
    DateFormatConfig("%H:%M %d/%m/%y", "HH:mm DD/MM/YY", observed_weight=0.02, target_weight=0.0),
    DateFormatConfig(
        "%I:%M%p %d/%m/%Y", "hh:mmp DD/MM/YYYY", observed_weight=0.02, target_weight=0.0
    ),
    DateFormatConfig("%H:%M %d/%m/%Y", "HH:mm DD/MM/YYYY", observed_weight=0.02, target_weight=0.0),
    DateFormatConfig(
        "%d/%m/%Y %I:%M%p", "DD/MM/YYYY hh:mmp", observed_weight=0.02, target_weight=0.0
    ),
    DateFormatConfig("%d/%m/%Y %H:%M", "DD/MM/YYYY HH:mm", observed_weight=0.02, target_weight=0.0),
    DateFormatConfig("%m/%d/%y", "MM/DD/YY", observed_weight=0.02, target_weight=0.0),
    DateFormatConfig(
        "%d/%m/%y %I:%M%p", "DD/MM/YY hh:mmp", observed_weight=0.02, target_weight=0.0
    ),
    DateFormatConfig("%d/%m/%y %H:%M", "DD/MM/YY HH:mm", observed_weight=0.02, target_weight=0.0),
]


def random_times_between(
    start: time.struct_time,
    end: time.struct_time,
    n: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> List[time.struct_time]:
    """Generate uniformly random datetimes between `start` and `end`

    Parameters
    ----------
    start :
        Start of the date/time window (Generate with e.g. `time.strptime()`).
    end :
        End of the date/time window (Generate with e.g. `time.strptime()`).
    n :
        Number of samples to generate.
    rng :
        Optional numpy random generator. Provide this to speed things up and enable reproducibility.

    Returns
    -------
    datetimes :
        List of `n` generated date/times in the given window. You can convert these to string
        representations via e.g. `time.strftime()`.
    """
    # Create a RNG if one was not provided:
    if rng is None:
        rng = np.random.default_rng()

    # To treat the struct_times as numeric (so we can add randomized offsets), convert them into
    # timestamps via mktime():
    start = time.mktime(start)
    end = time.mktime(end)

    # Generate random offsets as a 0-1 proportion through the window:
    props = rng.uniform(size=n)

    # localtime() is the inverse of mktime(), converting timestamps back to full time structs:
    max_offset = end - start
    return [time.localtime(start + p * max_offset) for p in props]


def generate_seq2seq_date_norm_dataset(
    n: int,
    configs: Sequence[DateFormatConfig] = DATE_FORMAT_CONFIGS,
    from_date: time.struct_time = time.strptime("1950-01-01", "%Y-%m-%d"),
    to_date: time.struct_time = time.strptime("2050-01-01", "%Y-%m-%d"),
    rng: Optional[np.random.Generator] = None,
) -> Dataset:
    """Generate a synthetic seq2seq task dataset for date normalization in text

    Parameters
    ----------
    n :
        Number of examples to generate
    configs :
        Sequence of date format configuration objects describing the date formats to use and their
        relative frequencies in source texts and target requests.
    from_date :
        Start of the date window that randomly generated dates should fall within.
    to_date :
        End of the date window that randomly generated dates should fall within.
    rng :
        Optional numpy random generator object. Provide this if you want reproducibility.

    Returns
    -------
    dataset :
        Hugging Face datasets.Dataset with fields `src_texts` (the input prompts) and `tgt_texts`
        (the target outputs) for each generated example.
    """
    # Create a RNG if one was not provided:
    if rng is None:
        rng = np.random.default_rng()

    # Normalize the observed_weights of the date format configurations:
    observed_weights = [fmt.observed_weight for fmt in configs]
    observed_weights_total = sum(observed_weights)
    if observed_weights_total != 1.0:
        logger.info(f"Normalizing observed_weights (summed to {observed_weights_total})")
        observed_weights = [w / observed_weights_total for w in observed_weights]
    # Select an observed format for the `n` input texts:
    obs_choices = rng.choice(
        len(observed_weights),
        p=observed_weights,
        size=(n,),
        replace=True,
    )

    # Normalize the target_weights of the date format configurations
    target_weights = [fmt.target_weight for fmt in configs]
    target_weights_total = sum(target_weights)
    if target_weights_total != 1.0:
        logger.info(f"Normalizing target_weights (summed to {target_weights_total})")
        target_weights = [w / target_weights_total for w in target_weights]
    # Select a requested format for the `n` prompts:
    target_choices = rng.choice(
        len(target_weights),
        p=target_weights,
        size=(n,),
        replace=True,
    )

    # Generate the `n` prompts & answers:
    random_dates = random_times_between(from_date, to_date, n=n, rng=rng)
    prompts = []
    answers = []
    for ix in range(n):
        obs_config = configs[obs_choices[ix]]
        target_config = configs[target_choices[ix]]
        random_date = random_dates[ix]
        prompt = "Convert dates to %s: %s" % (
            target_config.format_name,
            time.strftime(obs_config.format_str, random_date),
        )
        answer = time.strftime(target_config.format_str, random_date)
        prompts.append(prompt)
        answers.append(answer)

    return Dataset.from_dict(
        {
            "src_texts": prompts,
            "tgt_texts": answers,
        },
        info=DatasetInfo(
            description="Synthetic dataset for T5-style seq2seq normalization of dates",
        ),
    )
