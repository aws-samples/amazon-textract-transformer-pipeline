# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Notebook-simplifying utilities for working with Amazon S3
"""
# Python Built-Ins:
from typing import Optional, Tuple

# External Dependencies:
import boto3


s3client = boto3.client("s3")


def s3uri_to_bucket_and_key(s3uri: str) -> Tuple[str, str]:
    """Convert an s3://... URI string to a (bucket, key) tuple"""
    if not s3uri.lower().startswith("s3://"):
        raise ValueError(f"Expected S3 object URI to start with s3://. Got: {s3uri}")
    bucket, _, key = s3uri[len("s3://") :].partition("/")
    return bucket, key


def s3uri_to_relative_path(s3uri: str, key_base: str) -> str:
    """Extract e.g. 'subfolders/file' from 's3://bucket/.../{key_base}subfolders/file'

    If `key_base` is a folder, it should typically include a trailing slash.
    """
    return s3uri[len("s3://") :].partition("/")[2].partition(key_base)[2]


def s3_object_exists(bucket_name_or_s3uri: str, key: Optional[str] = None) -> bool:
    """Check if an object exists in Amazon S3

    Parameters
    ----------
    bucket_name_or_s3uri :
        Either an 's3://.../...' object URI, or an S3 bucket name.
    key :
        Ignored if `bucket_name_or_s3uri` is a full URI, otherwise mandatory: Key of the object to
        check.
    """
    if bucket_name_or_s3uri.lower().startswith("s3://"):
        bucket_name, key = s3uri_to_bucket_and_key(bucket_name_or_s3uri)
    elif not key:
        raise ValueError(
            "key is mandatory when bucket_name_or_s3uri is not an s3:// URI. Got: %s"
            % bucket_name_or_s3uri
        )
    else:
        bucket_name = bucket_name_or_s3uri
    try:
        s3client.head_object(Bucket=bucket_name, Key=key)
        return True
    except s3client.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        else:
            raise e
