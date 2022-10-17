# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import setuptools

with open("README.md") as fp:
    long_description = fp.read()

setuptools.setup(
    name="amazon-textract-transformer-pipeline",
    version="0.2.1",

    description="Post-processing Amazon Textract with Transformer-Based Models on Amazon SageMaker",
    long_description=long_description,
    long_description_content_type="text/markdown",

    author="Amazon Web Services",

    packages=["annotation", "pipeline"],

    install_requires=[
        "aws-cdk-lib==^2.46.0",
        "aws-cdk.aws-lambda-python-alpha==^2.46.0-alpha.0",
        "boto3==^1.24.0",
        "cdk-ecr-deployment==^2.5.6",
        "constructs==^10.0.101",
        "sagemaker>=2.87,<3",
    ],
    extras_require={
        "dev": [
            "black==^22.3.0",
            "black-nb==^0.7.0",
        ]
    },

    python_requires=">=3.8.0",

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
