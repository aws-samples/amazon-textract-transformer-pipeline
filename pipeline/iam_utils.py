# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CDK IAM convenience utilities for the OCR pipeline

Since we want to output a ManagedPolicy users can attach to their existing SageMaker execution
roles, but CDK ManagedPolicy objects do not implement IGrantable (see discussion at
https://github.com/aws/aws-cdk/issues/7448): The typical high-level access grant functions like
Bucket.grant_read_write() won't work for this use case and we'll instead define these utility
classes to simplify directly setting up useful IAM policies.
"""
# Python Built-Ins:
from itertools import zip_longest
from typing import Iterable, Union

# External Dependencies:
from aws_cdk import Stack
from aws_cdk.aws_iam import PolicyStatement
from aws_cdk.aws_s3 import Bucket
from aws_cdk.aws_ssm import IParameter
from aws_cdk.aws_stepfunctions import StateMachine


class S3Statement(PolicyStatement):
    """Utility class for creating PolicyStatement granting S3 read/write permissions"""

    def __init__(
        self,
        actions: Union[str, None] = None,
        grant_read: bool = True,
        grant_write: bool = False,
        resources: Iterable[Bucket] = [],
        resource_key_patterns: Iterable[str] = [],
        **kwargs,
    ):
        """Create a SsmParameterReadStatement

        Arguments
        ---------
        actions : Sequence[str] or None
            Appended to built-in list if provided
        grant_read : bool
            Whether to include built-in IAM actions for read permissions
        grant_write : bool
            Whether to include built-in IAM actions for write permissions
        resources : Iterable[s3.Bucket]
            S3 Buckets to grant read/write access to
        resource_key_patterns : Iterable[str]
            Key patterns to restrict access to, corresponding to the list in `resources`.
        **kwargs : Any
            Passed through to PolicyStatement
        """
        super().__init__(
            actions=(
                (["s3:GetBucket*", "s3:GetObject*", "s3:List*"] if grant_read else [])
                + (["s3:Abort*", "s3:DeleteObject*", "s3:PutObject*"] if grant_write else [])
                + (actions if actions else [])
            ),
            resources=[b.bucket_arn for b in resources]
            + [
                b.arn_for_objects(k or "*")
                for b, k in zip_longest(resources, resource_key_patterns)
            ],
            **kwargs,
        )


class SsmParameterReadStatement(PolicyStatement):
    """Utility class for creating PolicyStatement granting SSM parameter read permissions"""

    def __init__(
        self,
        actions: Union[str, None] = None,
        resources: Iterable[IParameter] = [],
        **kwargs,
    ):
        """Create a SsmParameterReadStatement

        Arguments
        ---------
        actions : Sequence[str] or None
            Appended to built-in list if provided
        resources : Iterable[ssm.IParameter]
            SSM parameters to grant read access to
        **kwargs : Any
            Passed through to PolicyStatement
        """
        super().__init__(
            actions=[
                "ssm:DescribeParameters",
                "ssm:GetParameter",
                "ssm:GetParameterHistory",
                "ssm:GetParameters",
            ]
            + (actions if actions else []),
            resources=[p.parameter_arn for p in resources],
            **kwargs,
        )


class SsmParameterWriteStatement(PolicyStatement):
    """Utility class for creating PolicyStatement granting SSM parameter read permissions"""

    def __init__(
        self,
        actions: Union[str, None] = None,
        resources: Iterable[IParameter] = [],
        **kwargs,
    ):
        """Create a SsmParameterWriteStatement

        Arguments
        ---------
        actions : Sequence[str] or None
            Appended to built-in list if provided
        resources : Iterable[ssm.IParameter]
            SSM parameters to grant read access to
        **kwargs : Any
            Passed through to PolicyStatement
        """
        super().__init__(
            actions=["ssm:PutParameter"] + (actions if actions else []),
            resources=[p.parameter_arn for p in resources],
            **kwargs,
        )


class StateMachineExecuteStatement(PolicyStatement):
    """Utility class for creating PolicyStatement granting execution of a SFn state machine"""

    def __init__(
        self,
        actions: Union[str, None] = None,
        resources: Iterable[StateMachine] = [],
        **kwargs,
    ):
        """Create a SsmParameterReadStatement

        Arguments
        ---------
        actions : Sequence[str] or None
            Appended to built-in list if provided
        resources : Iterable[sfn.StateMachine]
            SFn state machines to grant permissions on
        **kwargs : Any
            Passed through to PolicyStatement
        """
        super(StateMachineExecuteStatement, self).__init__(
            actions=[
                "states:DescribeExecution",
                "states:DescribeStateMachine",
                "states:DescribeStateMachineForExecution",
                "states:GetExecutionHistory",
                "states:ListExecutions",
                "states:StartExecution",
                "states:StartSyncExecution",
                "states:StopExecution",
            ]
            + (actions if actions else []),
            resources=[m.state_machine_arn for m in resources]
            + [
                "arn:{}:states:{}:{}:execution:{}:*".format(
                    Stack.of(m).partition,
                    Stack.of(m).region,
                    Stack.of(m).account,
                    m.state_machine_name,
                )
                for m in resources
            ],
            **kwargs,
        )
