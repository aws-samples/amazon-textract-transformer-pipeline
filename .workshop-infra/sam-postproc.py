#!/usr/bin/python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CLI tool to tokenize AWS regions in asset S3 URIs generated by AWS SAM

This enables easy cross-region deployment, by setting up cross-region-replicated asset hosting
buckets with region codes in the names.

You could probably do much the same thing in shell with a tool like jq, but Python gives lots of
flexibility to customize and extend where needed.
"""
# Python Built-Ins:
import argparse
import json
import re

AWS_REGION_SUFFIX_REGEX = (
    r"(?:af|ap|ca|eu|me|sa|us)-(?:central|north|south|(north|south)?east|(north|south)?west)-[1-3]$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Utility to parse SAM-generated JSON templates to multi-region-ify assets"
    )
    parser.add_argument(
        "infile",
        type=str,
        help="Path to input JSON template file generated by AWS SAM",
    )
    parser.add_argument(
        "outfile",
        type=str,
        help="Path to output file to save modified template",
    )
    return parser.parse_args()


def main(args):
    print(f"Loading input template... {args.infile}")
    with open(args.infile, "r") as fin:
        template = json.loads(fin.read())

    print("\nAdjusting region-suffixed asset URIs...")
    resources = template.get("Resources", {})
    n_edited = 0
    for resname in resources:
        resprops = resources[resname].get("Properties", {})
        for asset_attr in ("ContentUri", "CodeUri"):
            if (
                asset_attr in resprops
                and isinstance(resprops[asset_attr], str)
                and resprops[asset_attr].lower().startswith("s3://")
            ):
                bucket, _, key = resprops[asset_attr][len("s3://") :].partition("/")

                bucket_tokenized = re.sub(
                    AWS_REGION_SUFFIX_REGEX,
                    r"${AWS::Region}",
                    bucket,
                )
                if bucket != bucket_tokenized:
                    resprops[asset_attr] = {"Bucket": {"Fn::Sub": bucket_tokenized}, "Key": key}
                    n_edited += 1
                    print(f" - Region-tokenized {resname}.{asset_attr}")
    print(f"\nEdited {n_edited} resource properties\n")

    print(f"Writing output to {args.outfile}")
    with open(args.outfile, "w") as fout:
        fout.write(json.dumps(template, indent=2))


if __name__ == "__main__":
    args = parse_args()
    main(args)
