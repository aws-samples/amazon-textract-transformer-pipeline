# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import setuptools

with open("README.md") as fp:
    long_description = fp.read()

setuptools.setup(
    name="amazon-textract-transformer-pipeline",
    version="0.2.0",

    description="Post-processing Amazon Textract with Transformer-Based Models on Amazon SageMaker",
    long_description=long_description,
    long_description_content_type="text/markdown",

    author="Amazon Web Services",

    packages=["annotation", "pipeline"],

    install_requires=[
        "aws-cdk-lib==2.18.0",
        "aws-cdk.aws-lambda-python-alpha==2.18.0-alpha.0",
        "boto3==^1.17.92",
    ],
    extras_require={
        "dev": [
            "black==^21.6b0",
            "black-nb==^0.5.0",
        ]
    },

    python_requires=">=3.6.2",

    classifiers=[
        "Development Status :: 4 - Beta",

        "Intended Audience :: Developers",

        "License :: OSI Approved :: MIT No Attribution License (MIT-0)",

        "Programming Language :: JavaScript",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",

        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Utilities",

        "Typing :: Typed",
    ],
)
