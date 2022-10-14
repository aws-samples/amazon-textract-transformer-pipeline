# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Utilities for working with filesystem names and paths
"""
# Python Built-Ins:
import os
from typing import List, Tuple


def split_filename(filename: str) -> Tuple[str, str]:
    """Split a filename into base name and extension

    This basic method does NOT currently account for 2-part extensions e.g. '.tar.gz'
    """
    basename, _, ext = filename.rpartition(".")
    return basename, ext


def ls_relpaths(path: str, exclude_hidden: bool = True, sort: bool = True) -> List[str]:
    """Recursively list folder contents, sorting and excluding hidden files by default

    Parameters
    ----------
    path :
        Folder to be walked
    exclude_hidden :
        By default (True), exclude any files beginning with '.' or folders beginning with '.'
    sort :
        By default (True), sort result paths in alphabetical order. If False, results will be
        randomly ordered as per os.walk().

    Returns
    -------
    results :
        *Relative* file paths under the provided folder
    """
    if path.endswith("/"):
        path = path[:-1]
    result = [
        os.path.join(currpath, name)[len(path) + 1 :]  # +1 for trailing '/'
        for currpath, dirs, files in os.walk(path)
        for name in files
    ]
    if exclude_hidden:
        result = filter(
            lambda f: not (f.startswith(".") or "/." in f), result  # (Exclude hidden dot-files)
        )
    if sort:
        return sorted(result)
    else:
        return list(result)
