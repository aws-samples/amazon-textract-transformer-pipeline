# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CDK for deploying SageMaker Models / Endpoints
"""
# Python Built-Ins:
from logging import getLogger
import os
import subprocess
from typing import Dict, List, Mapping, Optional, Union

# External Dependencies:
from aws_cdk import BundlingOptions, BundlingOutput, IgnoreMode, Stack, SymlinkFollowMode
import aws_cdk.aws_ecr as ecr
from aws_cdk.aws_ecr_assets import (
    DockerImageAsset,
    DockerImageAssetInvalidationOptions,
    NetworkMode,
    Platform,
)
from aws_cdk.aws_iam import IRole, Policy, PolicyDocument, PolicyStatement
from aws_cdk.aws_kms import IKey
import aws_cdk.aws_lambda as aws_lambda
import aws_cdk.aws_s3 as s3
import aws_cdk.aws_s3_assets as s3assets
import aws_cdk.aws_sagemaker as sagemaker_cdk
from aws_cdk.aws_sns import ITopic
import cdk_ecr_deployment as imagedeploy
from constructs import Construct
from sagemaker.image_uris import retrieve as sm_image_retrieve


logger = getLogger("sagemaker_deployment")


class SageMakerDLCBasedImage(Construct):
    """Build (locally) and stage (to Amazon ECR) an image based on an AWS Deep Learning Container

    This construct is for use with local Dockerfiles that derive from SageMaker DLC bases -
    parameterized like:

    ```Dockerfile
    ARG BASE_IMAGE
    FROM ${BASE_IMAGE}
    ...
    ```

    You provide framework & version arguments similar to using sagemaker.image_uris.retrieve(). The
    construct looks up the base image URI via this same method; logs Docker in to the ECR registry;
    builds the container image locally (as a CDK DockerImageAsset); and pushes it to your ECR repo
    and tag of choice.
    """

    def __init__(
        self,
        scope: Construct,
        id: str,
        directory: str,
        ecr_repo: Union[str, ecr.IRepository],
        framework: str,
        use_gpu: bool,
        image_scope: str,
        py_version: str,
        version: str,
        build_args: Optional[Dict[str, str]] = None,
        exclude: Optional[List[str]] = None,
        extra_hash: Optional[str] = None,
        file: Optional[str] = None,
        follow_symlinks: Optional[SymlinkFollowMode] = None,
        ignore_mode: Optional[IgnoreMode] = None,
        invalidation: Optional[DockerImageAssetInvalidationOptions] = None,
        network_mode: Optional[NetworkMode] = None,
        platform: Optional[Platform] = None,
        target: Optional[str] = None,
        base_framework_version: Optional[str] = None,
        ecr_tag: str = "latest",
        **kwargs,
    ):
        """Create a SageMakerDLCBasedImage

        Parameters
        ----------
        scope :
            As per parent constructs.Construct
        id :
            As per parent constructs.Construct
        directory :
            Local folder build context (containing your custom Dockerfile).
        ecr_repo :
            Name of the Amazon ECR repository to stage the image to, or a CDK aws_ecr.Repository
            object. (Note that your image will *also* get pushed to a CDK-managed repository in
            your account by the CDK Asset bundling system).
        framework :
            Name of the base ML framework (as per SageMaker Python SDK image_uris.retrieve()).
        use_gpu :
            Set True to use the GPU-optimized version of the base image (if available), or False
            for the CPU-optimized version.
        image_scope :
            "training", "inference", etc (per SageMaker Python SDK image_uris.retrieve()).
        py_version :
            Python version for the base image (per SageMaker Python SDK image_uris.retrieve()).
        version :
            Framework version for the base image (per SageMaker Python SDK image_uris.retrieve()).
        build_args :
            Optional additional docker build args (other than BASE_IMAGE, which will be set
            automatically).
        exclude :
            File path patterns to exclude from docker build (per CDK DockerImageAsset).
        extra_hash :
            Optional extra information to encode into the fingerprint (per CDK DockerImageAsset).
        file :
            Optional explicit relative file path, if your Dockerfile is not named "Dockerfile" in
            the root of your `directory`.
        follow_symlinks :
            Symlink handling strategy for docker build (per CDK DockerImageAsset).
        ignore_mode :
            Ignore behaviour for `exclude` patterns (per CDK DockerImageAsset).
        invalidation :
            Options to control asset hash invalidation (as per CDK DockerImageAsset). Default: Hash
            all parameters.
        network_mode :
            Optional networking mode for RUN commands during build (per CDK DockerImageAsset).
        platform :
            Optional Docker target platform (per CDK DockerImageAsset).
        target :
            Optional Docker build target (per CDK DockerImageAsset).
        base_framework_version :
            Underlying ML framework version (per SageMaker Python SDK image_uris.retrieve(), as
            used for e.g. Hugging Face framework with both PyTorch and TensorFlow variants).
        ecr_tag :
            Tag name to use for your staged Amazon ECR image.
        """
        super().__init__(scope, id, **kwargs)

        # Look up the base container URI via SageMaker Python SDK (in whatever region):
        base_image_region = os.environ.get(
            "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        )
        base_image_args = {
            "framework": framework,
            "image_scope": image_scope,
            "instance_type": "ml.g4dn.xlarge" if use_gpu else "ml.c5.xlarge",
            "py_version": py_version,
            "region": base_image_region,
            "version": version,
        }
        if base_framework_version is not None:
            base_image_args["base_framework_version"] = base_framework_version
        base_image_uri = sm_image_retrieve(**base_image_args)

        # Pass the base image in as an arg for the (local) Docker build:
        if not build_args:
            build_args = {}
        build_args["BASE_IMAGE"] = base_image_uri

        # Since the DLCs live in AWS-managed accounts, we'll need to log in before local Docker can
        # access them:
        self.docker_ecr_login(base_image_uri)

        # Configure the target ECR repository, or create if a name was provided:
        if isinstance(ecr_repo, ecr.Repository):
            self.repo = ecr_repo
        else:
            self.repo = ecr.Repository.from_repository_name(self, "Repo", ecr_repo)

        image = DockerImageAsset(
            self,
            "Image",
            directory=directory,
            build_args=build_args,
            exclude=exclude,
            extra_hash=extra_hash,
            file=file,
            follow_symlinks=follow_symlinks,
            ignore_mode=ignore_mode,
            invalidation=invalidation,
            network_mode=network_mode,
            platform=platform,
            target=target,
        )

        self._dest_image = self.repo.repository_uri_for_tag(ecr_tag)
        self.deployment = imagedeploy.ECRDeployment(
            self,
            "Deployment",
            src=imagedeploy.DockerImageName(image.image_uri),
            dest=imagedeploy.DockerImageName(self._dest_image),
        )

    @property
    def image_uri(self) -> str:
        """URI of the staged Amazon ECR image"""
        return self._dest_image

    @classmethod
    def docker_ecr_login(cls, image_repo_or_host_uri: str) -> str:
        """Log the local docker agent in to target ECR repo/host via AWS CLI

        Parameters
        ----------
        image_repo_or_host_uri :
            URI of an Amazon ECR image, repository, or host to log in to

        Returns
        -------
        host_uri :
            Cleaned-up host URI that was logged in to: i.e. format like
            `{ACCOUNT_ID}.dkr.ecr.{REGION_NAME}.amazonaws.com`
        """
        ecr_host = image_repo_or_host_uri.partition("/")[0]
        region_name = ecr_host.split(".")[-3]

        login_cmd = " ".join(
            (
                "aws ecr get-login-password --region",
                region_name,
                "| docker login --username AWS --password-stdin",
                ecr_host,
            )
        )

        # Rather than just adding check=True to run(), capture output and check manually to ensure
        # we surface logs to console during CDK build:
        login_result = subprocess.run(
            login_cmd,
            shell=True,  # Need shell for the piping '|'
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if login_result.returncode != 0:
            logger.error(login_result.stdout.decode("utf-8"))
            raise subprocess.CalledProcessError(login_result.returncode, login_cmd)

        return ecr_host


class SageMakerCustomizedDLCModel(Construct):
    """Create a SageMaker Model based on a customized SageMakerDLCBasedImage

    This construct constructs a tarball from a local folder (similarly to using the SageMaker Python
    SDK), stages it to Amazon S3 (using the CDK assets bucket), and creates a SageMaker Model.

    It's intended mainly for "models" where inference code should be loaded in script mode, but
    there's no actual model weights/artifacts (like our pipeline's image pre-processing endpoint).
    """

    def __init__(
        self,
        scope: Construct,
        id: str,
        model_name: str,
        image: SageMakerDLCBasedImage,
        execution_role: IRole,
        entry_point: str,
        source_dir: Union[str, os.PathLike],
        environment: Optional[Mapping[str, str]] = None,
        max_payload_size: Optional[int] = 104857600,
    ):
        """Create a SageMakerCustomizedDLCModel

        Parameters
        ----------
        scope :
            As per CDK Construct parent.
        id :
            As per CDK Construct parent.
        model_name :
            SageMaker Model name to create in the API. (Currently mandatory).
        image :
            Customized SageMakerDLCBasedImage container the model should use. (Using pre-built
            SageMaker framework/algorithm here containers is not supported at this time).
        execution_role :
            IAM Role the Model should run with. Access to the tarball asset and container image
            will be automatically granted by this construct.
        entry_point :
            Relative path under `source_dir` to your model's inference script.
        source_dir :
            Local folder containing your model's inference script, requirements.txt, and any other
            required files.
        environment :
            Optional additional environment variables to set on your Model (SM script mode variables
            will be set by default).
        max_payload_size :
            By default, additional environment variables will be set to enable large request &
            response payload sizes within your container's serving stack. Your model will still be
            subject to SageMaker service payload limits (e.g. ~6MB if deployed to a real-time
            endpoint). This setting may not work for all frameworks as implemented (e.g. definitely
            not TensorFlow). Set a different number to adjust the limit, or None to remove these
            environment variables.
        """
        super().__init__(scope, id)

        self.asset = s3assets.Asset(
            self,
            "ModelTarball",
            path=source_dir,
            bundling=BundlingOptions(
                image=aws_lambda.Runtime.PYTHON_3_8.bundling_image,
                entrypoint=["bash", "-c"],
                # TODO: Nesting under code/ is good for PyTorch/HF, but not all frameworks
                command=["tar --transform 's,^,code/,' -czf /asset-output/model.tar.gz ."],
                output_type=BundlingOutput.ARCHIVED,
            ),
        )

        asset_bucket = s3.Bucket.from_bucket_name(self, "AssetBucket", self.asset.s3_bucket_name)
        assets_policy = Policy(
            self,
            "CDKArtifactAccess",
            document=PolicyDocument(
                statements=[
                    PolicyStatement(
                        actions=["s3:GetObject", "s3:GetObjectTorrent", "s3:GetObjectVersion"],
                        resources=[asset_bucket.arn_for_objects(self.asset.s3_object_key)],
                    ),
                    PolicyStatement(
                        actions=[
                            # These actions take no resources
                            "cloudwatch:PutMetricData",
                            "ecr:GetAuthorizationToken",
                            "logs:CreateLogDelivery",
                            "logs:DeleteLogDelivery",
                            "logs:GetLogDelivery",
                            "logs:ListLogDeliveries",
                            "logs:UpdateLogDelivery",
                        ],
                        resources=["*"],
                    ),
                    PolicyStatement(
                        actions=[
                            "ecr:BatchCheckLayerAvailability",
                            "ecr:BatchGetImage",
                            "ecr:GetDownloadUrlForLayer",
                            "ecr:ListImages",
                            "ecr:BatchCheckLayerAvailability",
                        ],
                        resources=[image.repo.repository_arn],
                    ),
                    PolicyStatement(
                        actions=[
                            # Log group level perms:
                            "logs:CreateLogGroup",
                            "logs:CreateLogStream",
                            # Log stream level perms:
                            "logs:PutLogEvents",
                        ],
                        # If you moved these perms from the model creation to the endpoint creation,
                        # could further scope them down to:
                        # Group: /aws/sagemaker/Endpoints/{ENDPOINT_NAME}
                        # Streams like "{VARIANTNAME}/{UNIQUE_ID}"
                        # arn:${Partition}:logs:${Region}:${Account}:log-group:${LogGroupName}
                        # arn:${Partition}:logs:${Region}:${Account}:log-group:${LogGroupName}:log-stream:${LogStreamName}
                        resources=["*"],
                    ),
                ],
            ),
            roles=[execution_role],
        )
        self._execution_role = execution_role

        env_final = {
            "PYTHONUNBUFFERED": "1",
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
            "SAGEMAKER_PROGRAM": entry_point,
            "SAGEMAKER_REGION": Stack.of(self).region,
            "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
            **(
                {}
                if max_payload_size is None
                else {
                    "MMS_MAX_REQUEST_SIZE": str(max_payload_size),
                    "MMS_MAX_RESPONSE_SIZE": str(max_payload_size),
                }
            ),
        }
        env_final.update(environment or {})

        self.cfn_model = sagemaker_cdk.CfnModel(
            self,
            "Model",
            execution_role_arn=execution_role.role_arn,
            primary_container=sagemaker_cdk.CfnModel.ContainerDefinitionProperty(
                environment=env_final,
                image=image.image_uri,
                model_data_url=self.asset.s3_object_url,
            ),
            model_name=model_name,
        )
        self.cfn_model.node.add_dependency(assets_policy)

    @property
    def execution_role(self) -> IRole:
        return self._execution_role

    @property
    def model_name(self) -> str:
        return self.cfn_model.model_name


class SageMakerAsyncInferenceConfig:
    """Immutable configuration class for SageMaker Async Inference

    The CfnEndpointConfig.AsyncInferenceConfigProperty class this configuration ultimately generates
    (via to_cfn_async_inference_config()) can be a bit limiting to work with in CDK, because e.g. it
    stores plain URI/ARN strings rather than underlying CDK constructs (for S3 buckets, SNS topics).

    This class stores original CDK construct inputs, but can still generate the final Cfn config.
    """

    def __init__(
        self,
        s3_output_bucket: s3.IBucket,
        s3_output_prefix: str = "",
        sns_success_topic: Optional[ITopic] = None,
        sns_error_topic: Optional[ITopic] = None,
        kms_key: Optional[IKey] = None,
        max_concurrent_invocations_per_instance: Optional[int] = None,
    ):
        """Create a SageMakerAsyncInferenceConfig

        Parameters
        ----------
        s3_output_bucket :
            s3.Bucket output location where inference results should be stored.
        s3_output_prefix :
            Prefix under `s3_output_bucket` where inference results should be stored.
        sns_success_topic :
            Optional SNS topic to receive notifications when an inference completes successfully.
        sns_error_topic :
            Optional SNS topic to receive notifications when an inference fails.
        kms_key :
            KMS key if required for S3 output
        max_concurrent_invocations_per_instance :
            Limit on concurrent requests per instance in the endpoint, to manage resource use.
        """
        self._s3_output_bucket = s3_output_bucket
        self._s3_output_prefix = s3_output_prefix
        self._sns_success_topic = sns_success_topic
        self._sns_error_topic = sns_error_topic
        self._kms_key = kms_key
        self._max_concurrent = max_concurrent_invocations_per_instance

    def to_cfn_async_inference_config(
        self,
    ) -> sagemaker_cdk.CfnEndpointConfig.AsyncInferenceConfigProperty:
        """Realise the final CfnEndpointConfig.AsyncInferenceConfigProperty from this config"""
        if self._sns_success_topic:
            success_topic_arn = self._sns_success_topic.topic_arn
        else:
            success_topic_arn = None
        if self._sns_error_topic:
            error_topic_arn = self._sns_error_topic.topic_arn
        else:
            error_topic_arn = None
        if success_topic_arn or error_topic_arn:
            notification_config = (
                sagemaker_cdk.CfnEndpointConfig.AsyncInferenceNotificationConfigProperty(
                    error_topic=error_topic_arn,
                    success_topic=success_topic_arn,
                )
            )
        else:
            notification_config = None

        return sagemaker_cdk.CfnEndpointConfig.AsyncInferenceConfigProperty(
            client_config=sagemaker_cdk.CfnEndpointConfig.AsyncInferenceClientConfigProperty(
                max_concurrent_invocations_per_instance=self._max_concurrent,
            ),
            output_config=sagemaker_cdk.CfnEndpointConfig.AsyncInferenceOutputConfigProperty(
                s3_output_path=self._s3_output_bucket.s3_url_for_object(self._s3_output_prefix),
                kms_key_id=self._kms_key.key_id if self._kms_key else None,
                notification_config=notification_config,
            ),
        )

    @property
    def s3_output_bucket(self) -> s3.IBucket:
        return self._s3_output_bucket

    @property
    def s3_output_prefix(self) -> str:
        return self._s3_output_prefix

    @property
    def sns_error_topic(self) -> Optional[ITopic]:
        return self._sns_error_topic

    @property
    def sns_success_topic(self) -> Optional[ITopic]:
        return self._sns_success_topic


class SageMakerModelDeployment(Construct):
    """Deploy a SageMaker Model to an Endpoint (real-time or async)

    This construct provides a simplified interface for setting up a SageMaker Endpoint from a
    Model in CDK, somewhat like the SageMaker Python SDK's model.deploy() function.

    SNS and S3 IAM permissions will be automatically added to the model's execution role if needed
    for async inference.
    """

    def __init__(
        self,
        scope: Construct,
        id: str,
        endpoint_name: str,
        model: SageMakerCustomizedDLCModel,
        instance_type: str,
        initial_instance_count: int = 1,
        async_inference_config: Optional[SageMakerAsyncInferenceConfig] = None,
    ):
        """Create a SageMakerModelDeployment

        Parameters
        ----------
        scope :
            As per CDK Construct parent.
        id :
            As per CDK Construct parent.
        endpoint_name :
            Name of the SageMaker Endpoint to create (mandatory). An Endpoint Configuration will be
            created using the same name.
        model :
            Existing SageMakerCustomizedDLCModel construct to use for the endpoint.
        instance_type :
            Instance type to use in endpoint deployment, e.g. "ml.c5.xlarge" or "ml.g4dn.xlarge".
        initial_instance_count :
            Initial number of instances to deploy for the endpoint. (Note this construct doesn't yet
            provide settings for auto-scaling).
        async_inference_config :
            Provide this configuration if you'd like to deploy an asynchronous endpoint. Default
            `None` yields a real-time endpoint.
        """
        super().__init__(scope, id)

        # Ensure model IAM role gets sufficient S3 and SNS permissions for async deployment:
        if async_inference_config is not None:
            async_inference_config.s3_output_bucket.grant_read_write(
                model.execution_role,
                async_inference_config.s3_output_prefix + "/*",
            )

            if async_inference_config.sns_success_topic is not None:
                async_inference_config.sns_success_topic.grant_publish(model.execution_role)
                if (async_inference_config.sns_error_topic is not None) and (
                    async_inference_config.sns_error_topic
                    != async_inference_config.sns_success_topic
                ):
                    async_inference_config.sns_error_topic.grant_publish(model.execution_role)

        self.endpoint_config = sagemaker_cdk.CfnEndpointConfig(
            self,
            "EndpointConfig",
            endpoint_config_name=endpoint_name,
            async_inference_config=async_inference_config.to_cfn_async_inference_config(),
            # data_capture_config not yet supported
            production_variants=[
                sagemaker_cdk.CfnEndpointConfig.ProductionVariantProperty(
                    initial_variant_weight=1.0,
                    model_name=model.model_name,
                    variant_name="AllTraffic",
                    initial_instance_count=initial_instance_count,
                    instance_type=instance_type,
                    # serverless_config not yet supported
                )
            ],
        )
        self.endpoint_config.add_depends_on(model.cfn_model)

        self.endpoint = sagemaker_cdk.CfnEndpoint(
            self,
            "Endpoint",
            endpoint_config_name=self.endpoint_config.endpoint_config_name,
            endpoint_name=endpoint_name,
        )
        self.endpoint.add_depends_on(self.endpoint_config)
