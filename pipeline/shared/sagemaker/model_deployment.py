# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CDK for deploying SageMaker Models / Endpoints
"""
# Python Built-Ins:
from dataclasses import dataclass
from enum import Enum
from logging import getLogger
import os
import subprocess
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

# External Dependencies:
from aws_cdk import (
    AssetHashType,
    BundlingOptions,
    BundlingOutput,
    CfnTag,
    Duration,
    IgnoreMode,
    RemovalPolicy,
    Stack,
    SymlinkFollowMode,
)
import aws_cdk.aws_applicationautoscaling as appscaling
from aws_cdk.aws_cloudwatch import Metric
import aws_cdk.aws_ecr as ecr
from aws_cdk.aws_ecr_assets import (
    DockerImageAsset,
    DockerImageAssetInvalidationOptions,
    NetworkMode,
    Platform,
)
import aws_cdk.aws_iam as iam
from aws_cdk.aws_kms import IKey
import aws_cdk.aws_lambda as aws_lambda
import aws_cdk.aws_s3 as s3
import aws_cdk.aws_s3_assets as s3assets
import aws_cdk.aws_sagemaker as sagemaker_cdk
from aws_cdk.aws_sns import ITopic
import cdk_ecr_deployment as imagedeploy
from constructs import Construct
from sagemaker.image_uris import retrieve as sm_image_retrieve
from semver import Version


logger = getLogger("sagemaker_deployment")


class ModelServerType(str, Enum):
    """Enumeration of known base model server types

    SageMaker inference DLCs for different frameworks/versions may be based on different underlying
    model server implementations. This enum represents which family of model server is used, for
    code that tries to set serving configuration variables.
    """

    MMS = "MMS"  # (AWS Labs Multi-Model Server)
    TFSERVING = "TFServing"
    TORCHSERVE = "TorchServe"
    UNKNOWN = "Unknown"


@dataclass
class SageMakerDLCSpec:
    """Data class to specify a SageMaker Deep Learning Container

    Contains parameters for looking up container image URIs via SageMaker Python SDK.

    Parameters
    ----------
    framework :
        Name of the base ML framework (as per SageMaker Python SDK image_uris.retrieve()).
    py_version :
        Python version for the base image (per SageMaker Python SDK image_uris.retrieve()).
    version :
        Framework version for the base image (per SageMaker Python SDK image_uris.retrieve()).
    use_gpu :
        Set True to use the GPU-optimized version of the base image (if available), or False
        for the CPU-optimized version.
    image_scope :
        "training", "inference", etc (per SageMaker Python SDK image_uris.retrieve()).
    base_framework_version :
        Underlying ML framework version (per SageMaker Python SDK image_uris.retrieve(), as
        used for e.g. Hugging Face framework with both PyTorch and TensorFlow variants).
    """

    framework: str
    py_version: str
    version: str
    use_gpu: bool = False
    image_scope: str = "inference"
    base_framework_version: Optional[str] = None

    def model_server_type(self) -> ModelServerType:
        """Infer from the configuration what kind of model serving stack the image uses

        This information is useful for code that wants to configure throughput/etc management
        options on the serving stack. The function may return ModelServerType.UNKNOWN. It's also
        not exhaustively checked - may be incorrect for some older framework versions.

        Raises
        ------
        ValueError :
            If called on a non-inference scoped image (e.g. a training image)
        """
        if self.image_scope != "inference":
            raise ValueError(
                "Model server type is only applicable to 'inference' images, not '%s'"
                % (self.image_scope,)
            )
        if self.framework == "huggingface":
            return ModelServerType.MMS

        if self.framework == "pytorch":
            if self.semver() >= "1.6.0":
                return ModelServerType.TORCHSERVE
            else:
                return ModelServerType.MMS
        if self.framework == "tensorflow":
            if self.semver() >= "1.13.0":
                return ModelServerType.TFSERVING
            else:
                return ModelServerType.MMS
        return ModelServerType.UNKNOWN

    def semver(self) -> Version:
        """Normalize the `version` field to a semver library semantic Version object

        Raises
        ------
        ValueError :
            If the `version` cannot be parsed as expected
        """
        # semver library requires full semantic versions, but SMSDK sometimes works with shorthand
        # Major.Minor (no .Patch). Check and add a .0 if needed to keep it happy:
        raw_ver_components = len(self.version.split("."))
        if raw_ver_components < 3:
            semver_str = ".".join([self.version] + (["0"] * (3 - raw_ver_components)))
        else:
            semver_str = self.version
        return Version.parse(semver_str)

    def to_sm_image_retrieve_args(self, region_name: str) -> dict:
        """Generate keyword arguments for sagemaker.image_uris.retrieve() URI lookup"""
        args = {
            "framework": self.framework,
            "image_scope": self.image_scope,
            "instance_type": "ml.g4dn.xlarge" if self.use_gpu else "ml.c5.xlarge",
            "py_version": self.py_version,
            "region": region_name,
            "version": self.version,
        }
        if self.base_framework_version is not None:
            args["base_framework_version"] = self.base_framework_version
        return args


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
        base_image_spec: SageMakerDLCSpec,
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
        base_image_spec :
            Parameters describing the SageMaker Deep Learning Container image to use as a base.
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
        ecr_tag :
            Tag name to use for your staged Amazon ECR image.
        """
        super().__init__(scope, id, **kwargs)
        self._base_image_spec = base_image_spec

        # Look up the base container URI via SageMaker Python SDK (in whatever region):
        base_image_region = os.environ.get(
            "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        )
        base_image_uri = sm_image_retrieve(
            **base_image_spec.to_sm_image_retrieve_args(region_name=base_image_region)
        )

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
            self.repo = ecr.Repository(
                self,
                "Repo",
                image_scan_on_push=True,
                removal_policy=RemovalPolicy.DESTROY,
                repository_name=ecr_repo,
            )

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

    def build_inference_config_env(
        self,
        max_payload_bytes: Optional[int] = 100 * 1024 * 1024,
        timeout_secs: Optional[int] = 15 * 60,
    ) -> Dict[str, str]:
        """Build environment dictionary to configure the serving stack

        The environment variables defined by this function configure the *in-container* serving
        stack. They won't help you exceed SageMaker-side limits on payload size or time-out, but
        may be important for e.g. async endpoints where the SageMaker-side limits are high but the
        default model server limits are lower.

        Parameters
        ----------
        max_payload_bytes :
            Maximum request/response payload (both set together by this function) size the model
            should allow, in bytes. Set `None` to omit the environment variable and use the
            container's default setting (which is usually lower).
        timeout_secs :
            Number of seconds the model server should allow for processing before timing out. Set
            `None` to omit the environment variable and use the container's default setting (which
            is usually lower).

        Returns
        -------
        environment :
            Str->str dictionary of environment variables to set on the SageMaker Model

        Raises
        ------
        ValueError :
            If this is a non-inference image, or the model server type of the image is unknown
        NotImplementedError :
            If the model server type is known, but this function can't configure it yet
        """
        server_type = self._base_image_spec.model_server_type()
        if server_type is ModelServerType.UNKNOWN:
            raise ValueError(
                "Can't configure inference because this container's model server type is unknown"
            )

        if server_type is ModelServerType.MMS:
            var_prefix = "MMS_"
        elif server_type is ModelServerType.TORCHSERVE:
            var_prefix = "TS_"
        else:
            raise NotImplementedError(
                f"Inference server configuration is not yet supported for {server_type.value}"
            )

        result = {}
        if timeout_secs is not None:
            result[f"{var_prefix}DEFAULT_RESPONSE_TIMEOUT"] = str(timeout_secs)
        if max_payload_bytes is not None:
            result[f"{var_prefix}MAX_REQUEST_SIZE"] = str(max_payload_bytes)
            result[f"{var_prefix}MAX_RESPONSE_SIZE"] = str(max_payload_bytes)
        return result


class ModelTarballAsset(s3assets.Asset):
    """A CDK S3 asset to tarball a local folder for use as a SageMaker model artifact

    Nests the source folder under a 'code/' prefix in the output tarball, for consistency with
    PyTorch and HuggingFace framework container expectations.
    """

    def __init__(
        self,
        scope: Construct,
        id: str,
        *,
        source_dir: str,
        readers: Optional[Sequence[iam.IGrantable]] = None,
        asset_hash: Optional[str] = None,
        asset_hash_type: Optional[AssetHashType] = None,
        exclude: Optional[Sequence[str]] = None,
        follow_symlinks: Optional[SymlinkFollowMode] = None,
        ignore_mode: Optional[IgnoreMode] = None,
    ) -> None:
        super().__init__(
            scope,
            id,
            path=source_dir,
            readers=readers,
            asset_hash=asset_hash,
            asset_hash_type=asset_hash_type,
            bundling=BundlingOptions(
                image=aws_lambda.Runtime.PYTHON_3_9.bundling_image,
                entrypoint=["bash", "-c"],
                # TODO: Nesting under code/ is good for PyTorch/HF, but not all frameworks
                command=["tar --transform 's,^,code/,' -czf /asset-output/model.tar.gz ."],
                output_type=BundlingOutput.ARCHIVED,
            ),
            exclude=exclude,
            follow_symlinks=follow_symlinks,
            ignore_mode=ignore_mode,
        )


class SageMakerEndpointExecutionRole(iam.Role):
    """IAM Role with base permissions to use for running SageMaker inference endpoints.

    This class automatically sets up a default inline policy granting permissions to store logs and
    metrics, as well as pull access to ECR repositories you specify.
    """

    def __init__(
        self,
        scope: Construct,
        id: str,
        *,
        description: Optional[str] = None,
        external_ids: Optional[Sequence[str]] = None,
        inline_policies: Optional[Mapping[str, iam.PolicyDocument]] = None,
        managed_policies: Optional[Sequence[iam.IManagedPolicy]] = None,
        max_session_duration: Optional[Duration] = None,
        path: Optional[str] = None,
        permissions_boundary: Optional[iam.IManagedPolicy] = None,
        role_name: Optional[str] = None,
        ecr_repositories: List[ecr.IRepository] = [],
    ):
        """Create a SageMakerEndpointExecutionRole

        Parameters as per aws_iam.Role except where explicitly specified.

        Parameters
        ----------
        ecr_repositories :
            List of Amazon ECR repositories that this role should be able to pull container images
            from. You can add more after construction via `add_ecr_repo()`.
        """
        super().__init__(
            scope,
            id,
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description=description,
            external_ids=external_ids,
            inline_policies=inline_policies,
            managed_policies=managed_policies,
            max_session_duration=max_session_duration,
            path=path,
            permissions_boundary=permissions_boundary,
            role_name=role_name,
        )

        ecr_read_statement = iam.PolicyStatement(
            actions=[
                "ecr:BatchCheckLayerAvailability",
                "ecr:BatchGetImage",
                "ecr:GetDownloadUrlForLayer",
                "ecr:ListImages",
                "ecr:BatchCheckLayerAvailability",
            ],
            resources=ecr_repositories[:],
        )
        s3_read_statement = iam.PolicyStatement(
            actions=["s3:GetObject", "s3:GetObjectTorrent", "s3:GetObjectVersion"],
            resources=[],
        )

        # We create the policy as a specific construct (rather than just incorporating into
        # inline_policies) so it can be added to centrally and explicitly depended on.
        default_policy = iam.Policy(
            self,
            "DefaultSMPolicy",
            document=iam.PolicyDocument(
                statements=[
                    ecr_read_statement,
                    s3_read_statement,
                    iam.PolicyStatement(
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
                    iam.PolicyStatement(
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
            roles=[self],
        )
        self._default_policy = default_policy
        self._ecr_read_statement = ecr_read_statement
        self._s3_read_statement = s3_read_statement

    def add_ecr_repo(self, repository: ecr.Repository) -> None:
        """Add permission to pull images from the given repo to this role's inline policy"""
        self._ecr_read_statement.add_resources(repository.repository_arn)

    def add_s3_asset(self, asset: s3assets.Asset) -> None:
        """Add permission to read an S3 asset to this role's inline policy"""
        self._s3_read_statement.add_resources(
            s3.Bucket.from_bucket_name(
                self,
                "AssetBucket",
                asset.s3_bucket_name,
            ).arn_for_objects("*"),
        )

    @property
    def default_policy(self) -> iam.Policy:
        """Reference to this role's default inline policy (in case you need to dependOn it)"""
        return self._default_policy


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
        image: SageMakerDLCBasedImage,
        execution_role: SageMakerEndpointExecutionRole,
        entry_point: str,
        source_dir: Union[str, os.PathLike],
        environment: Optional[Mapping[str, str]] = None,
        max_payload_size: Optional[int] = 104857600,
        max_response_secs: Optional[int] = 60 * 15,
        model_name: Optional[str] = None,
        tags: Optional[Sequence[Union[CfnTag, Dict[str, Any]]]] = None,
    ):
        """Create a SageMakerCustomizedDLCModel

        Parameters
        ----------
        scope :
            As per CDK Construct parent.
        id :
            As per CDK Construct parent.
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
            endpoint). This setting may not work for all frameworks as currently implemented. Set a
            different number to adjust the limit, or set both this and `max_response_secs` to `None`
            to skip this config for unsupported frameworks.
        max_response_secs :
            By default, additional environment variables will be set to enable long response times
            within your container's serving stack. Your model will still be subject to SageMaker
            service timeout limits. This setting may not work for all frameworks as currently
            implemented. Set a different number to adjust the limit, or set both this and
            `max_payload_size` to `None` to skip this config for unsupported frameworks.
        model_name :
            Optional explicit SageMaker Model name to create in the API.
        tags :
            Optional resource tags to apply to the generated Model.
        """
        super().__init__(scope, id)

        self.asset = ModelTarballAsset(
            self,
            "ModelTarball",
            source_dir=source_dir,
            # Adding the execution_role as a 'reader' here is not sufficient as the generated policy
            # can't be depended on by the CFnModel below.
        )
        execution_role.add_s3_asset(self.asset)
        execution_role.add_ecr_repo(image.repo)
        self._execution_role = execution_role

        env_final = {
            "PYTHONUNBUFFERED": "1",
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
            "SAGEMAKER_PROGRAM": entry_point,
            "SAGEMAKER_REGION": Stack.of(self).region,
            "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
            **(
                {}
                if max_payload_size is None and max_response_secs is None
                else image.build_inference_config_env(
                    max_payload_bytes=max_payload_size,
                    timeout_secs=max_response_secs,
                )
            ),
        }
        env_final.update(environment or {})

        self._cfn_model = sagemaker_cdk.CfnModel(
            self,
            "Model",
            execution_role_arn=execution_role.role_arn,
            primary_container=sagemaker_cdk.CfnModel.ContainerDefinitionProperty(
                environment=env_final,
                image=image.image_uri,
                model_data_url=self.asset.s3_object_url,
            ),
            model_name=model_name,
            tags=tags,
        )
        self._cfn_model.node.add_dependency(execution_role.default_policy)
        self._cfn_model.node.add_dependency(image.deployment)

    @property
    def execution_role(self) -> iam.IRole:
        return self._execution_role

    @property
    def model_name(self) -> str:
        return self._cfn_model.attr_model_name


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


class SageMakerAutoscalingRole(iam.Role):
    """IAM role with service principal and inline policies pre-configured for SM auto-scaling"""

    def __init__(
        self,
        scope: Construct,
        id: str,
        *,
        description: Optional[str] = None,
        external_ids: Optional[Sequence[str]] = None,
        inline_policies: Optional[Mapping[str, iam.PolicyDocument]] = None,
        managed_policies: Optional[Sequence[iam.IManagedPolicy]] = None,
        max_session_duration: Optional[Duration] = None,
        path: Optional[str] = None,
        permissions_boundary: Optional[iam.IManagedPolicy] = None,
        role_name: Optional[str] = None,
        autoscaling_policy_name: str = "SageMakerAutoscaling",
        **kwargs,
    ):
        """Create a SageMakerAutoscalingRole

        The role principal is set automatically. An inline policy will automatically be added (with
        name given by `autoscaling_policy_name`) to grant autoscaling permissions. Otherwise,
        parameters are as per `aws_iam.Role`.
        """
        if not inline_policies:
            inline_policies = {}
        if autoscaling_policy_name in inline_policies:
            raise ValueError(
                "Cannot create autoscaling_policy_name %s because this name is already taken in "
                "the provided inline_policies map." % autoscaling_policy_name
            )
        inline_policies[autoscaling_policy_name] = iam.PolicyDocument(
            statements=[
                iam.PolicyStatement(
                    actions=[
                        "autoscaling:*",
                        "sagemaker:DescribeEndpoint",
                        "sagemaker:DescribeEndpointConfig",
                        "sagemaker:UpdateEndpointWeightsAndCapacities",
                        "cloudwatch:PutMetricAlarm",
                        "cloudwatch:DescribeAlarms",
                        "cloudwatch:DeleteAlarms",
                    ],
                    resources=["*"],
                ),
                iam.PolicyStatement(
                    actions=["iam:CreateServiceLinkedRole"],
                    resources=[
                        "arn:aws:iam::*:role/aws-service-role/sagemaker.application-autoscaling.amazonaws.com/AWSServiceRoleForApplicationAutoScaling_SageMakerEndpoint"
                    ],
                    conditions={
                        "StringLike": {
                            "iam:AWSServiceName": "sagemaker.application-autoscaling.amazonaws.com",
                        }
                    },
                ),
            ],
        )

        super().__init__(
            scope,
            id,
            assumed_by=iam.ServicePrincipal("sagemaker.application-autoscaling.amazonaws.com"),
            description=description,
            external_ids=external_ids,
            inline_policies=inline_policies,
            managed_policies=managed_policies,
            max_session_duration=max_session_duration,
            path=path,
            permissions_boundary=permissions_boundary,
            role_name=role_name,
            **kwargs,
        )


class EndpointAutoscaler(appscaling.ScalableTarget):
    """Auto-scaling target for a SageMaker endpoint

    Use standard methods like `scale_to_track_metric()` and `scale_on_metric()` to define actual
    scaling policies as usual. In addition to the pre-defined target metric from
    `appscaling.PredefinedMetric.SAGEMAKER_VARIANT_INVOCATIONS_PER_INSTANCE`, you can use the
    convenience methods on this class to specify other common, custom SageMaker metrics for async
    inference: `sagemaker_backlog_size_metric()` and `sagemaker_backlog_without_cap_metric()`.
    """

    def __init__(
        self,
        scope: Construct,
        id: str,
        *,
        max_capacity: int,
        min_capacity: int,
        endpoint_name: str,
        variant_name: str = "AllTraffic",
        role: Optional[iam.Role] = None,
        **kwargs,
    ):
        """Create an EndpointAutoscaler

        Parameters as per aws_applicationautoscaling.ScalableTarget with the following exceptions:
        - Instead of a `resource_id`, you specify a SageMaker `endpoint_name` and a `variant_name`
            (which defaults to the standard 'AllTraffic' variant name if not set)
        - You do not need to specify the `service_namespace` or `scalable_dimension`
        - If you don't provide an IAM Role, one will be created for you with SageMaker endpoint
            auto-scaling permissions.
        """
        self._endpoint_name = endpoint_name
        self._variant_name = variant_name
        resource_id = f"endpoint/{endpoint_name}/variant/{variant_name}"
        if role is None:
            role = SageMakerAutoscalingRole(scope, "AutoScalingRole")
        super().__init__(
            scope,
            id,
            max_capacity=max_capacity,
            min_capacity=min_capacity,
            resource_id=resource_id,
            scalable_dimension="sagemaker:variant:DesiredInstanceCount",
            service_namespace=appscaling.ServiceNamespace.SAGEMAKER,
            role=role,
            **kwargs,
        )

    @classmethod
    def sagemaker_backlog_size_metric_for_endpoint(cls, endpoint_name: str) -> Metric:
        return Metric(
            metric_name="ApproximateBacklogSizePerInstance",
            namespace="AWS/SageMaker",
            dimensions_map={
                "EndpointName": endpoint_name,
            },
            statistic="Average",
        )

    def sagemaker_backlog_size_metric(self) -> Metric:
        return self.sagemaker_backlog_size_metric_for_endpoint(self._endpoint_name)

    @classmethod
    def sagemaker_backlog_without_cap_metric_for_endpoint(cls, endpoint_name: str) -> Metric:
        return Metric(
            metric_name="HasBacklogWithoutCapacity",
            namespace="AWS/SageMaker",
            dimensions_map={
                "EndpointName": endpoint_name,
            },
            statistic="Average",
        )

    def sagemaker_backlog_without_cap_metric(self) -> Metric:
        return self.sagemaker_backlog_without_cap_metric_for_endpoint(self._endpoint_name)

    def scale_async_endpoint_simple(
        self,
        target_backlog_per_instance: float = 5.0,
        tracking_scale_in_cooldown: Duration = Duration.minutes(5),
        tracking_scale_out_cooldown: Duration = Duration.minutes(5),
        bootstrap_cooldown: Duration = Duration.seconds(150),
    ) -> List[Union[appscaling.StepScalingPolicy, appscaling.TargetTrackingScalingPolicy]]:
        """Create default scaling policies for a responsive but economical SageMaker async endpoint

        Set up scaling policies to track a target average backlog size per instance (typically >1),
        but also to spin up at least one instance when a single inference request is received. If
        backlog target tracking is used alone, no instances would be started until the backlog
        threshold is exceeded.

        Parameters
        ----------
        target_backlog_per_instance :
            Overall target request queue length per instance for auto-scaling (useful for
            high-volume scaling).
        tracking_scale_in_cooldown :
            Cooldown period for reducing capacity due to insufficient backlog
        tracking_scale_out_cooldown :
            Cooldown period for adding capacity due to excessive backlog
        bootstrap_cooldown :
            Cooldown period for adding a single "bootstrap" instance when a request arrives and no
            instances are currently runnning.
        """
        tracking_policy = self.scale_to_track_metric(
            "TargetScaling",
            target_value=target_backlog_per_instance,
            custom_metric=self.sagemaker_backlog_size_metric(),
            scale_in_cooldown=tracking_scale_in_cooldown,
            scale_out_cooldown=tracking_scale_out_cooldown,
        )
        bootstrap_policy = self.scale_on_metric(
            "BootstrapScaling",
            metric=self.sagemaker_backlog_without_cap_metric(),
            scaling_steps=[
                appscaling.ScalingInterval(change=1, lower=1.0),
                # Dummy step, never triggered but required for param validation:
                appscaling.ScalingInterval(change=0, upper=-10),
            ],
            adjustment_type=appscaling.AdjustmentType.CHANGE_IN_CAPACITY,
            cooldown=bootstrap_cooldown,
        )
        return [tracking_policy, bootstrap_policy]


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
        model: SageMakerCustomizedDLCModel,
        instance_type: str,
        initial_instance_count: int = 1,
        async_inference_config: Optional[SageMakerAsyncInferenceConfig] = None,
        endpoint_name: Optional[str] = None,
        tags: Optional[Sequence[Union[CfnTag, Dict[str, Any]]]] = None,
    ):
        """Create a SageMakerModelDeployment

        Parameters
        ----------
        scope :
            As per CDK Construct parent.
        id :
            As per CDK Construct parent.
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
        endpoint_name :
            Optional explicit of the SageMaker Endpoint to create. An Endpoint Configuration will be
            created using the same name, if set.
        tags :
            Optional resource tags (to be applied to both SageMaker EndpointConfig and Endpoint)
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

        self._endpoint_config = sagemaker_cdk.CfnEndpointConfig(
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
            tags=tags,
        )
        self._endpoint_config.add_depends_on(model._cfn_model)

        self._endpoint = sagemaker_cdk.CfnEndpoint(
            self,
            "Endpoint",
            endpoint_config_name=self._endpoint_config.attr_endpoint_config_name,
            endpoint_name=endpoint_name,
            tags=tags,
        )
        self._endpoint.add_depends_on(self._endpoint_config)

    @property
    def endpoint_name(self) -> str:
        return self._endpoint.attr_endpoint_name
