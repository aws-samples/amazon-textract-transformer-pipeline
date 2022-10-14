# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""ML project infrastructure utilities (reading stack params from AWS SSM)

init() with a valid *project ID* (or provide the PROJECT_ID environment variable), and this module
will read the project's configuration from AWS SSM (stored there by the CloudFormation stack):
Allowing us to reference from the SageMaker notebook to resources created by the stack.

A "session" in the context of this module is a project config loaded from SSM. This way we can
choose either to init-and-forget (standard data science project sandbox use case) or to call
individual functions on separate sessions (interact with multiple projects).
"""

# Python Built-Ins:
import logging
import os
from types import SimpleNamespace
from typing import Dict, Union

# External Dependencies:
import boto3

logger = logging.getLogger("project")
ssm = boto3.client("ssm")


defaults = SimpleNamespace()
defaults.project_id = None
defaults.session = None


if "PROJECT_ID" not in os.environ:
    logger.info(
        "No PROJECT_ID variable found in environment: You'll need to call init('myprojectid')"
    )
else:
    defaults.project_id = os.environ["PROJECT_ID"]


class ProjectSession:
    """Class defining the parameters for a project and how they get loaded (from AWS SSSM)"""

    SSM_PREFIX: str = ""
    STATIC_PARAMS: Dict[str, str] = {
        "static/A2IExecutionRoleArn": "a2i_execution_role_arn",
        "static/InputBucket": "pipeline_input_bucket_name",
        "static/ReviewsBucket": "pipeline_reviews_bucket_name",
        "static/PipelineStateMachine": "pipeline_sfn_arn",
        "static/PlainTextractStateMachine": "plain_textract_sfn_arn",
        "static/PreprocImageURI": "preproc_image_uri",
        "static/ThumbnailsCallbackTopicArn": "thumbnails_callback_topic_arn",
        "static/ModelCallbackTopicArn": "model_callback_topic_arn",
        "static/ModelResultsBucket": "model_results_bucket",
        "static/SMDockerBuildRole": "sm_image_build_role",
    }
    DYNAMIC_PARAM_NAMES: Dict[str, str] = {
        "config/HumanReviewFlowArn": "a2i_review_flow_arn_param",
        "config/EntityConfiguration": "entity_config_param",
        "config/SageMakerEndpointName": "sagemaker_endpoint_name_param",
        "config/ThumbnailEndpointName": "thumbnail_endpoint_name_param",
    }

    # Static values (from SSM):
    a2i_execution_role_arn: str
    pipeline_input_bucket_name: str
    pipeline_reviews_bucket_name: str
    pipeline_sfn_arn: str
    plain_textract_sfn_arn: str
    preproc_image_uri: str
    model_callback_topic_arn: str
    model_results_bucket: str
    sm_image_build_role: str
    thumbnails_callback_topic_arn: str
    # Configurable parameters (names in SSM):
    a2i_review_flow_arn_param: str
    entity_config_param: str
    sagemaker_endpoint_name_param: str
    thumbnail_endpoint_name_param: str

    def __init__(self, project_id: str):
        """Create a ProjectSession

        Parameters
        ----------
        project_id :
            ProjectId from the provisioned OCR pipeline stack
        """
        self.project_id = project_id

        # Load SSM names for dynamic/configuration parameters:
        for param_suffix, session_attr in self.DYNAMIC_PARAM_NAMES.items():
            setattr(self, session_attr, f"{self.SSM_PREFIX}/{project_id}/{param_suffix}")

        # Load SSM *values* for static project attributes:
        param_names_to_config_attrs = {
            f"{self.SSM_PREFIX}/{project_id}/{s}": self.STATIC_PARAMS[s] for s in self.STATIC_PARAMS
        }
        response = ssm.get_parameters(Names=[s for s in param_names_to_config_attrs])
        n_invalid = len(response.get("InvalidParameters", []))
        if n_invalid == len(param_names_to_config_attrs):
            raise ValueError(f"Found no valid SSM parameters for /{project_id}: Invalid project ID")
        elif n_invalid > 0:
            logger.warning(
                " ".join(
                    [
                        f"{n_invalid} Project parameters missing from SSM: Some functionality ",
                        f"may not work as expected. Missing: {response['InvalidParameters']}",
                    ]
                )
            )

        for param in response["Parameters"]:
            param_name = param["Name"]
            setattr(self, param_names_to_config_attrs[param_name], param["Value"])

    def __repr__(self):
        """Produce a meaningful representation when this class is print()ed"""
        typ = type(self)
        mod = typ.__module__
        qualname = typ.__qualname__
        propdict = self.__dict__
        proprepr = ",\n  ".join([f"{k}={propdict[k]}" for k in propdict])
        return f"<{mod}.{qualname}(\n  {proprepr}\n) at {hex(id(self))}>"


def init(project_id: str) -> ProjectSession:
    """Initialise the project util library (and the default session) to project_id"""
    # Check that we can create the session straight away, for nice error behaviour:
    session = ProjectSession(project_id)
    if defaults.project_id and defaults.project_id != project_id and defaults.session:
        logger.info(f"Clearing previous default session for project '{defaults.project_id}'")
    defaults.project_id = project_id
    defaults.session = session
    logger.info(f"Working in project '{project_id}'")
    return session


def session_or_default(sess: Union[ProjectSession, None] = None):
    """Mostly-internal utility to return either the provided session or else a default"""
    if sess:
        return sess
    elif defaults.session:
        return defaults.session
    elif defaults.project_id:
        defaults.session = ProjectSession(defaults.project_id)
        return defaults.session
    else:
        raise ValueError(
            "Must provide a project session or init() the project library with a valid project ID"
        )
