# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Utilities for working with SageMaker Ground Truth
"""

# Python Built-Ins:
import json
from logging import getLogger
import os
from string import Template
from textwrap import dedent
from typing import Iterable, Optional

# External Dependencies:
import boto3  # General-purpose AWS SDK for Python

# Local Dependencies:
from .postproc.config import FieldConfiguration

botosess = boto3.Session()
s3 = botosess.resource("s3")
smclient = botosess.client("sagemaker")

logger = getLogger("smgt")


# Lambda function ARN components for pre- and post-processing with built-in task types, as per:
# https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_HumanTaskConfig.html
# https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_AnnotationConsolidationConfig.html
SMGT_LAMBDA_CONFIG = {
    "bounding-box": {
        "fn-name-pre": "PRE-BoundingBox",
        "fn-name-post": "ACS-BoundingBox",
        "regions": {
            "us-east-1": {"account-id": "432418664414"},
            "us-east-2": {"account-id": "266458841044"},
            "us-west-2": {"account-id": "081040173940"},
            "ca-central-1": {"account-id": "918755190332"},
            "eu-west-1": {"account-id": "568282634449"},
            "eu-west-2": {"account-id": "487402164563"},
            "eu-central-1": {"account-id": "203001061592"},
            "ap-northeast-1": {"account-id": "477331159723"},
            "ap-northeast-2": {"account-id": "845288260483"},
            "ap-south-1": {"account-id": "565803892007"},
            "ap-southeast-1": {"account-id": "377565633583"},
            "ap-southeast-2": {"account-id": "454466003867"},
        },
    },
    "bounding-box-adjustment": {
        "fn-name-pre": "PRE-AdjustmentBoundingBox",
        "fn-name-post": "ACS-AdjustmentBoundingBox",
        "regions": {
            "us-east-1": {"account-id": "432418664414"},
            "us-east-2": {"account-id": "266458841044"},
            "us-west-2": {"account-id": "081040173940"},
            "ca-central-1": {"account-id": "918755190332"},
            "eu-west-1": {"account-id": "568282634449"},
            "eu-west-2": {"account-id": "487402164563"},
            "eu-central-1": {"account-id": "203001061592"},
            "ap-northeast-1": {"account-id": "477331159723"},
            "ap-northeast-2": {"account-id": "845288260483"},
            "ap-south-1": {"account-id": "565803892007"},
            "ap-southeast-1": {"account-id": "377565633583"},
            "ap-southeast-2": {"account-id": "454466003867"},
        },
    },
}

# Template for built-in bounding box task type, as per:
# https://docs.aws.amazon.com/sagemaker/latest/dg/sms-bounding-box.html
# ...with placeholders for initial value and instructions added.
BBOX_TEMPLATE = """<script src="https://assets.crowd.aws/crowd-html-elements.js"></script>
<crowd-form>
  <crowd-bounding-box
    name="boundingBox"
    src="{{ task.input.taskObject | grant_read_access }}"
    header="${header}"
    ${initial_value_attr}
    labels="{{ task.input.labels | to_json | escape }}"
  >

    <full-instructions header="Bounding box instructions">${instructions_full}</full-instructions>
  
    <short-instructions>${instructions_short}</short-instructions>
  
  </crowd-bounding-box>
</crowd-form>
"""

# Liquid template to incorporate initial bbox values from an existing manifest field, as per:
# https://docs.aws.amazon.com/sagemaker/latest/dg/sms-ui-template-crowd-bounding-box.html
BBOX_INITIAL_VALUE_TEMPLATE = """initial-value="[
  {% for box in task.input.manifestLine.${label_attribute_name}.annotations %}
    {% capture class_id %}{{ box.class_id }}{% endcapture %}
    {% assign label = task.input.manifestLine.${label_attribute_name}-metadata.class-map[class_id] %}
    {
      label: {{label | to_json}},
      left: {{box.left}},
      top: {{box.top}},
      width: {{box.width}},
      height: {{box.height}},
    },
  {% endfor %}
]"
"""


def ensure_bucket_cors(
    bucket_name: str,
    aws_account_id: str = os.environ.get("AWS_ACCOUNT_ID"),
) -> Optional[dict]:
    """Ensure S3 bucket has a GET * CORS rule as required by SageMaker Ground Truth

    Parameters
    ----------
    bucket_name :
        Name of the S3 bucket to configure. You must have s3:GetBucketCors and s3:PutBucketCors
        permissions on this bucket.
    aws_account_id :
        AWS Account ID. If not provided, will attempt to determine from AWS_ACCOUNT_ID environment
        variable.

    Returns
    -------
    resp :
        An S3 PutBucketCors response if a rule was added, else None if a rule was already
        present.
    """
    bucket_cors = s3.BucketCors(bucket_name)

    try:
        existing_rules = bucket_cors.cors_rules
    except s3.meta.client.exceptions.ClientError as err:
        if err.response.get("Error", {}).get("Code") == "NoSuchCORSConfiguration":
            existing_rules = []
        else:
            raise err

    if any(
        r for r in existing_rules if "*" in r["AllowedOrigins"] and "GET" in r["AllowedMethods"]
    ):
        logger.info(f"Bucket already set up with CORS permissions: %s", bucket_name)
        return None
    else:
        new_rules = existing_rules + [
            {
                "ID": "SageMakerGroundTruth",
                "AllowedHeaders": [],
                "AllowedMethods": ["GET"],
                "AllowedOrigins": ["*"],
                "ExposeHeaders": [],
                "MaxAgeSeconds": 60,
            },
        ]
        cors_resp = bucket_cors.put(
            CORSConfiguration={"CORSRules": new_rules},
            ExpectedBucketOwner=aws_account_id,
        )
        logger.info("Added CORS permissions to bucket: %s", bucket_name)
        return cors_resp


def get_smgt_lambda_arn(
    pre: bool,
    task: str = "bounding-box",
    region: Optional[str] = None,
) -> str:
    """Get the pre- or post-processing Lambda ARN for a SM Ground Truth built-in task type

    Based on documentation from:
    https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_HumanTaskConfig.html
    https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_AnnotationConsolidationConfig.html

    Parameters
    ----------
    pre : bool
        True to return the pre-processing ARN, False to return the post-processing ARN
    task : str
        Which built-in task you're looking for
    region : Optional[str]
        The AWS Region you're working in (if not set, will auto-discover from boto3).
    """
    if task not in SMGT_LAMBDA_CONFIG:
        raise NotImplementedError(
            f"Task type '{task}' is not in known set {[k for k in SMGT_LAMBDA_CONFIG]}"
        )
    task_cfg = SMGT_LAMBDA_CONFIG[task]
    if not region:
        region = botosess.region_name

    if region not in task_cfg["regions"]:
        raise NotImplementedError(f"Lambda details not known for task {task} in region {region}")

    return "arn:aws:lambda:{}:{}:function:{}".format(
        region,
        task_cfg["regions"][region]["account-id"],
        task_cfg["fn-name-pre" if pre else "fn-name-post"],
    )


def generate_label_category_config(
    field_configs: Iterable[FieldConfiguration],
    reviewing_attribute_name: Optional[str] = None,
) -> dict:
    """Generate content for 'label configuration file' for an SMGT annotation job on pre-built UI

    The label configuration file includes parameters like the list of class labels and the worker
    instructions, which need to be populated when using the pre-built task UIs. For more info on
    file structure, see:

    https://docs.aws.amazon.com/sagemaker/latest/dg/sms-label-cat-config-attributes.html#sms-label-cat-config-attributes-schema

    Parameters
    ----------
    field_configs : Iterable[FieldConfiguration]
        List of field configuration objects describing the label classes (see postproc utils)
    reviewing_attribute_name : Optional[str]
        If setting up a label review/validation job, provide the name of the attribute in the
        manifest where previous annotations are stored.
    """
    result = {
        "document-version": "2018-11-28",
        "labels": [{"label": field.name} for field in field_configs],
        "instructions": {
            "shortInstruction": (
                "Draw bounding boxes to highlight all instances of the listed entity types. Use "
                "overlapping boxes of the same type to highlight contiguous, non-square regions. "
                "See the full instructions for more details."
            ),
            "fullInstruction": dedent(
                """
                <p>Use the bounding box tool to highlight <em>all instances</em> of the listed
                entity types on the page.</p>

                <p>Overlapping boxes of the same type are consolidated to a single object by the
                model: So you can use this pattern to highlight <em>non-rectangular regions</em>
                (such as specific multi-line sentences in a paragraph); and should <em>avoid</em>
                overlapping same-class boxes where the two regions are semantically separate.</p>
                """
            ),
        },
    }
    if any(field.annotation_guidance for field in field_configs):
        guidance = ["<h3>Per-Field Guidance</h3>"]
        for field in filter(lambda f: f.annotation_guidance, field_configs):
            guidance.append(f"<h4>{field.name}</h4>")
            guidance.append(field.annotation_guidance)
        result["instructions"]["fullInstruction"] += "\n".join(guidance)

    if reviewing_attribute_name is not None:
        result["auditLabelAttributeName"] = reviewing_attribute_name

    return result


def get_bbox_template(
    header: str,
    instructions_short: str = "",
    instructions_full: str = "",
    reviewing_attribute_name: Optional[str] = None,
) -> str:
    initial_value_attr = (
        Template(BBOX_INITIAL_VALUE_TEMPLATE).substitute(
            {"label_attribute_name": reviewing_attribute_name}
        )
        if reviewing_attribute_name
        else ""
    )
    return Template(BBOX_TEMPLATE).substitute(
        {
            "header": header,
            "initial_value_attr": initial_value_attr,
            "instructions_short": instructions_short,
            "instructions_full": instructions_full,
        }
    )


def workteam_arn_from_name(name: str) -> str:
    """Validate that a SageMaker Ground Truth Workteam exists with members, and return its ARN"""
    desc = smclient.describe_workteam(WorkteamName=name)
    if not len(desc["Workteam"]["MemberDefinitions"]):
        raise ValueError(f"Workteam '{name}' has no members! Add members to use for annotation")

    return desc["Workteam"]["WorkteamArn"]


def create_bbox_labeling_job(
    job_name: str,
    bucket_name: str,
    execution_role_arn: str,
    fields: Iterable[FieldConfiguration],
    input_manifest_s3uri: str,
    output_s3uri: str,
    workteam_arn: str,
    local_inputs_folder: str = os.path.join("data", "manifests"),
    reviewing_attribute_name: Optional[str] = None,
    s3_inputs_prefix: str = "data/manifests",
    task_template: Optional[str] = None,
    pre_lambda_arn: Optional[str] = None,
    post_lambda_arn: Optional[str] = None,
) -> dict:
    """Create a SageMaker Ground Truth labelling job with the built-in Bounding Box task UI

    Parameters
    ----------
    job_name :
        Name of the job to create (must be unique in your AWS Account+Region)
    bucket_name :
        Name of the S3 bucket where input/output manifests and job metadata will be stored
    execution_role_arn :
        ARN of the SageMaker Execution Role (in AWS IAM) that the labelling job will run as. The
        role must have permission to access your selected `bucket_name`.
    fields : Iterable[FieldConfiguration]
        Field/entity types list
    input_manifest_s3uri :
        's3://...' URI where the input JSON-Lines manifest file is (already) stored
    output_s3uri :
        's3://...' URI where the job output should be stored (SMGT will add a job subfolder)
    workteam_arn :
        ARN of the SageMaker Ground Truth workteam who will be performing the task
    local_inputs_folder :
        Local folder where configuration files for SMGT will be stored before uploading to S3.
        (Default 'data/manifests')
    reviewing_attribute_name : Optional[str]
        Set the name of the manifest attribute where existing labels are stored, to trigger an
        adjustment job on pre-existing labels. (Default None)
    s3_inputs_prefix :
        Key prefix (with or without trailing slash) under which configuration files for SMGT will
        be uploaded to the S3 bucket_name. (Default 'data/manifests')
    task_template :
        Optional custom task template file (local path). If not provided, the standard SMGT Bounding
        Box task UI will be used.
    pre_lambda_arn :
        Override AWS Lambda ARN for Ground Truth task pre-processing. When unset, the default
        pre-processing Lambda for SMGT Bounding Box (adjustment) task UI will be used. Set this
        parameter to use your own function instead.
    post_lambda_arn :
        Override AWS Lambda ARN for Ground Truth task post-processing. When unset, the default
        post-processing Lambda for SMGT Bounding Box (adjustment) task UI will be used. Set this
        parameter to use your own function instead.

    Returns
    -------
    response : dict
        As per boto3 sagemaker client.create_labeling_job()
    """
    # Validate/normalize inputs:
    if local_inputs_folder.endswith(os.path.sep):
        local_inputs_folder = local_inputs_folder[:-1]
    if s3_inputs_prefix.startswith("/"):
        s3_inputs_prefix = s3_inputs_prefix[1:]
    if s3_inputs_prefix.endswith("/"):
        s3_inputs_prefix = s3_inputs_prefix[:-1]
    bucket = s3.Bucket(bucket_name)

    # Generate and upload a job metadata file (including things like the list of class names, and
    # any instructions to include for workers):
    input_category_file = os.path.join(local_inputs_folder, f"{job_name}.meta.json")
    input_category_s3key = "/".join((s3_inputs_prefix, f"{job_name}.meta.json"))
    input_category_s3uri = f"s3://{bucket_name}/{input_category_s3key}"
    with open(input_category_file, "w") as f:
        label_category_config = generate_label_category_config(
            fields,
            reviewing_attribute_name=reviewing_attribute_name,
        )
        f.write(json.dumps(label_category_config))
    bucket.upload_file(input_category_file, input_category_s3key)
    print(f"Uploaded Labeling Category Config {input_category_file} to:\n{input_category_s3uri}")

    # Generate and upload the task template:
    task_template_s3key = "/".join((s3_inputs_prefix, f"{job_name}.liquid.html"))
    task_template_s3uri = f"s3://{bucket_name}/{task_template_s3key}"
    if task_template is None:
        task_template_file = os.path.join(local_inputs_folder, f"{job_name}.liquid.html")
        with open(task_template_file, "w") as f:
            f.write(
                get_bbox_template(
                    header="Highlight the entities with bounding boxes",
                    instructions_short=(
                        label_category_config.get("instructions", {}).get("shortInstruction", "")
                    ),
                    instructions_full=(
                        label_category_config.get("instructions", {}).get("fullInstruction", "")
                    ),
                    reviewing_attribute_name=reviewing_attribute_name,
                )
            )
    else:
        task_template_file = task_template
    bucket.upload_file(task_template_file, task_template_s3key)
    print(f"Uploaded resolved task UI template {task_template_file} to:\n{task_template_s3uri}")

    # Create the actual labeling job:
    task = "bounding-box-adjustment" if reviewing_attribute_name else "bounding-box"
    return smclient.create_labeling_job(
        LabelingJobName=job_name,
        LabelAttributeName=job_name,
        InputConfig={
            "DataSource": {
                "S3DataSource": {"ManifestS3Uri": input_manifest_s3uri},
            },
            # If adapting this code for use with A2I public workforce, you may need to add
            # additional content classifiers as described here:
            # https://docs.aws.amazon.com/sagemaker/latest/dg/sms-workforce-management-public.html
            # https://docs.aws.amazon.com/augmented-ai/2019-11-07/APIReference/API_HumanLoopDataAttributes.html
            # "DataAttributes": {
            #     "ContentClassifiers": [
            #         "FreeOfPersonallyIdentifiableInformation", "FreeOfAdultContent"
            #     ],
            # },
        },
        OutputConfig={
            "S3OutputPath": output_s3uri,
        },
        RoleArn=execution_role_arn,
        LabelCategoryConfigS3Uri=input_category_s3uri,  # Required for built-in tasks only
        HumanTaskConfig={
            "WorkteamArn": workteam_arn,
            "UiConfig": {
                "UiTemplateS3Uri": task_template_s3uri,
            },
            "PreHumanTaskLambdaArn": (
                get_smgt_lambda_arn(pre=True, task=task)
                if pre_lambda_arn is None
                else pre_lambda_arn
            ),
            "TaskTitle": "Credit Card Agreement Entities",
            "TaskDescription": "Highlight the entities with bounding boxes",
            "NumberOfHumanWorkersPerDataObject": 1,
            "TaskTimeLimitInSeconds": 60 * 60,
            "TaskAvailabilityLifetimeInSeconds": 10 * 24 * 60 * 60,
            "MaxConcurrentTaskCount": 250,
            "AnnotationConsolidationConfig": {
                "AnnotationConsolidationLambdaArn": (
                    get_smgt_lambda_arn(pre=False, task=task)
                    if post_lambda_arn is None
                    else post_lambda_arn
                ),
            },
        },
    )
