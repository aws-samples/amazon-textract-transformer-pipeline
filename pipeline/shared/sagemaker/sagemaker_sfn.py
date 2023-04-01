# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CDK construct for using a dynamic SageMaker endpoint from AWS Step Functions

While Step Functions already integrates with SageMaker, the Lambda function and associated SFn task
constructs in this folder provide some additional functionality including as dynamically retrieving
the endpoint name from SSM, or transparently handling sync vs async endpoints - without requiring
additional SFn steps.
"""
# Python Built-Ins:
from typing import List, Mapping, Optional, Union

# External Dependencies:
from aws_cdk import Duration
from aws_cdk.aws_iam import Effect, PolicyStatement, Role
from aws_cdk.aws_lambda import Runtime as LambdaRuntime
from aws_cdk.aws_lambda_python_alpha import PythonFunction
import aws_cdk.aws_sns as sns
import aws_cdk.aws_sns_subscriptions as subs
import aws_cdk.aws_ssm as ssm
import aws_cdk.aws_stepfunctions as sfn
import aws_cdk.aws_stepfunctions_tasks as sfn_tasks
from constructs import Construct
import jsii

# Local Dependencies:
from .. import abs_path

SM_LAMBDA_PATH = abs_path("fn-call-sagemaker", __file__)


class SageMakerCallerFunction(PythonFunction):
    """Construct for a SageMaker-Step Functions integration Lambda function

    This function may be created per-point-of-use in Step Functions, or shared between different
    `SageMakerSSMStep`s so long as the EndpointName(Param) is specified at the point of use.

    See fn-call-sagemaker/main.py docstring for more information on features and SFn state I/O.
    """

    def __init__(
        self,
        scope: Construct,
        id: str,
        support_async_endpoints: bool,
        *args,
        default_endpoint_name_param: Optional[ssm.StringParameter] = None,
        sid_prefix: Union[str, None] = "",
        environment: Optional[Mapping[str, str]] = None,
        memory_size: Optional[jsii.Number] = 128,
        timeout: Optional[Duration] = Duration.seconds(60),
        **kwargs,
    ) -> None:
        """Create a SageMakerCallerFunction

        Parameters
        ----------
        support_async_endpoints :
            Set true to enable support for async SageMaker endpoints in the function, or False to
            support *only* synchronous (real-time) endpoints.
        default_endpoint_name_param :
            Optional SSM StringParameter specifying the default SageMaker endpoint name to invoke.
        sid_prefix :
            Optional prefix for generated IAM statement IDs on the function's role, to prevent
            overlap if the role is shared. Set `None` to disable setting SIDs altogether. Default
            "" will produce standard SIDs.
        **kwargs :
            Other parameters are as per CDK PythonFunction, with some modifications to defaults.
        """
        self._support_async_endpoints = support_async_endpoints
        default_environment = {"SUPPORT_ASYNC_ENDPOINTS": "1" if support_async_endpoints else "0"}
        self._default_endpoint_name_param = default_endpoint_name_param
        if default_endpoint_name_param:
            environment["DEFAULT_ENDPOINT_NAME_PARAM"] = default_endpoint_name_param.parameter_name

        environment = {**default_environment, **(environment if environment else {})}
        super().__init__(
            scope,
            id,
            *args,
            entry=SM_LAMBDA_PATH,
            runtime=LambdaRuntime.PYTHON_3_9,
            handler="handler",
            index="main.py",
            environment=environment,
            memory_size=memory_size,
            timeout=timeout,
            **kwargs,
        )

        self.add_to_role_policy(
            PolicyStatement(
                sid=(
                    None
                    if sid_prefix is None
                    else (sid_prefix + "SageMakerEndpointsDescribeInvoke")
                ),
                actions=["sagemaker:InvokeEndpoint"]
                + (
                    ["sagemaker:DescribeEndpoint", "sagemaker:InvokeEndpointAsync"]
                    if support_async_endpoints
                    else []
                ),
                effect=Effect.ALLOW,
                resources=["*"],
            )
        )
        if support_async_endpoints:
            self.add_to_role_policy(
                PolicyStatement(
                    sid=None if sid_prefix is None else (sid_prefix + "StateMachineNotify"),
                    actions=["states:SendTask*"],
                    effect=Effect.ALLOW,
                    resources=["*"],  # No resource types currently supported per the doc
                )
            )

    @property
    def default_endpoint_name_param(self):
        return self._default_endpoint_name_param

    @property
    def support_async_endpoints(self):
        return self._support_async_endpoints


class SageMakerSSMStep(sfn_tasks.LambdaInvoke):
    """Step Functions task to invoke an SSM-set SageMaker (sync or async) endpoint via Lambda"""

    async_notify_topic: Optional[sns.Topic]
    lambda_function: SageMakerCallerFunction

    def __init__(
        self,
        scope: Construct,
        id: str,
        lambda_role: Role,
        *args,
        support_async_endpoints: bool = True,
        async_notify_topic: Optional[sns.ITopic] = None,
        lambda_function: Optional[SageMakerCallerFunction] = None,
        default_endpoint_name_param: Optional[ssm.StringParameter] = None,
        **kwargs,
    ) -> None:
        """Create a SageMakerSSMStep

        Parameters
        ----------
        lambda_role :
            Existing IAM role to use for the AWS Lambda function
        support_async_endpoints :
            Set true to enable support for async SageMaker endpoints in the function, or False to
            support *only* synchronous (real-time) endpoints. (Default True)
        async_notify_topic :
            Optional pre-existing SNS Topic to use for async inference callbacks (if this has been
            pre-created). Note: Sharing topics between SageMakerSSMSteps is *not* recommended.
        lambda_function :
            Optional pre-existing `SageMakerCallerFunction` to use (useful for sharing one Lambda
            between multiple SageMaker SFn task steps)
        default_endpoint_name_param :
            Optional SSM StringParameter specifying the default SageMaker endpoint name to invoke.
        **kwargs :
            Other parameters are as per CDK SFn LambdaInvoke.
        """
        if lambda_function:
            if lambda_function.support_async_endpoints != support_async_endpoints:
                raise ValueError(
                    (
                        "Provided caller_lambda's support_async_endpoints setting ({}) is "
                        "different from this step construct's setting ({}). Either use the same "
                        "setting in each construct, or let this construct create its own caller "
                        "Lambda function."
                    ).format(lambda_function.support_async_endpoints, support_async_endpoints)
                )
            if default_endpoint_name_param and (
                lambda_function.default_endpoint_name_param != default_endpoint_name_param
            ):
                raise ValueError(
                    (
                        "Provided caller_lambda's default_endpoint_name_param setting ({}) is "
                        "different from this step construct's setting ({}). Either use the same "
                        "setting in each construct, or let this construct create its own caller "
                        "Lambda function."
                    ).format(
                        lambda_function.default_endpoint_name_param, default_endpoint_name_param
                    )
                )
        else:
            lambda_function = SageMakerCallerFunction(
                self,
                "CallSageMaker",
                support_async_endpoints=support_async_endpoints,
                default_endpoint_name_param=default_endpoint_name_param,
                role=lambda_role,
            )

        for k in ("integration_pattern", "payload_response_only"):
            if k in kwargs:
                raise ValueError(
                    f"sfn_tasks.LambdaInvoke parameter {k} is set internally by SageMakerSSMStep"
                )

        super().__init__(
            scope,
            id,
            *args,
            lambda_function=lambda_function,
            integration_pattern=(
                sfn.IntegrationPattern.WAIT_FOR_TASK_TOKEN
                if support_async_endpoints
                else sfn.IntegrationPattern.REQUEST_RESPONSE
            ),
            payload_response_only=None if support_async_endpoints else True,
            **kwargs,
        )

        self.lambda_function = lambda_function
        self.support_async_endpoints = support_async_endpoints
        if support_async_endpoints:
            # Need to customize topic name because of: https://github.com/aws/aws-cdk/issues/7832
            self.async_notify_topic = async_notify_topic or sns.Topic(self, f"SageMakerAsync-{id}")
            self.async_notify_topic.add_subscription(subs.LambdaSubscription(lambda_function))
        else:
            self.async_notify_topic = async_notify_topic

    def sagemaker_sns_statements(self, sid_prefix: Union[str, None] = "") -> List[PolicyStatement]:
        """Create PolicyStatements to grant SageMaker permission to use the SNS callback topic

        Arguments
        ---------
        sid_prefix : str | None
            Prefix to add to generated statement IDs for uniqueness, or "", or None to suppress
            SIDs.
        """
        if not self.async_notify_topic:
            return []

        return [
            PolicyStatement(
                actions=["sns:Publish"],
                effect=Effect.ALLOW,
                resources=[self.async_notify_topic.topic_arn],
                sid=None if sid_prefix is None else (sid_prefix + "PublishSageMakerTopic"),
            )
        ]
