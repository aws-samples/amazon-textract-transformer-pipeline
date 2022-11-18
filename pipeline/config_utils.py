#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Utilities for configuring the stack (e.g. environment variable parsing)
"""
# Python Built-Ins:
import os
from typing import List, Optional


def bool_env_var(env_var_name: str, default: Optional[bool] = None) -> bool:
    """Parse a boolean environment variable

    Raises
    ------
    ValueError :
        If environment variable `env_var_name` is not found and no `default` is specified, or if the
        raw value string could not be interpreted as a boolean.

    Returns
    -------
    parsed :
        True if the env var has values such as `1`, `true`, `y`, `yes` (case-insensitive). False if
        opposite values `0`, `false`, `n`, `no` or empty string.
    """
    raw = os.environ.get(env_var_name)
    if raw is None:
        if default is None:
            raise ValueError(f"Mandatory boolean env var '{env_var_name}' not found")
        return default
    raw = raw.lower()
    if raw in ("1", "true", "y", "yes"):
        return True
    elif raw in ("", "0", "false", "n", "no"):
        return False
    else:
        raise ValueError(
            "Couldn't interpret env var '%s' as boolean. Got: '%s'" % (env_var_name, raw)
        )


def list_env_var(env_var_name: str, default: Optional[List[str]] = None) -> List[str]:
    """Parse a comma-separated string list from an environment variable

    Raises
    ------
    ValueError :
        If environment variable `env_var_name` is not found and no `default` is specified.

    Returns
    -------
    parsed :
        List of strings: Split by commas in the raw input, each stripped of any leading/trailing
        whitespace, and filtered to remove any empty values. For example: `dog, , cat` returns
        `["dog", "cat"]`. Empty environment variable returns `[]`. Whitespace stripping and
        filtering is not applied to the `default` value, if used.
    """
    raw = os.environ.get(env_var_name)
    if raw is None:
        if default is None:
            raise ValueError(f"Mandatory string-list env var {env_var_name} not found")
        return default[:]
    whitespace_stripped_values = [s.strip() for s in raw.split(",")]
    return [s for s in whitespace_stripped_values if s]
