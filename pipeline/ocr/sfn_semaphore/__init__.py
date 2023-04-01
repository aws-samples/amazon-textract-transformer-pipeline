# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CDK Constructs for concurrency + rate control via Step Functions and DynamoDB semaphore locks

Usage:

- Set up a DynamoDB table to store your concurrency locks. You can use SFnSemaphoreDynamoDbTable
  for nice configuration defaults. Tables can be shared between multiple semaphores so long as each
  semaphore's lock name (DDB item ID) is unique.
- Set up a SFnSemaphoreReaper to clear potentially leaked locks in cases where your state machine
  fails. Reapers can also be shared between multiple semaphores, if you like.
- Define the `sfn.Chain` of worker states you'd like to limit the concurrency & entry rate of.
- Create a SFnSemaphore to wrap your workchain with lock acquisition & release steps.
- Use the `SFnSemaphore.chain` in your state machine definition (maybe alongside other steps that
  don't need concurrency control)
- `.attach()` your SFnSemaphoreReaper to your state machine, to make sure leaked locks are cleaned
  up if errors or exceptions prevent normal lock release.
"""
# Python Built-Ins:
from __future__ import annotations
from typing import Optional

# External Dependencies:
from aws_cdk import Duration, RemovalPolicy
import aws_cdk.aws_dynamodb as dynamodb
import aws_cdk.aws_events as aws_events
import aws_cdk.aws_events_targets as events_targets
from aws_cdk.aws_iam import ServicePrincipal
from aws_cdk.aws_lambda import Runtime as LambdaRuntime
from aws_cdk.aws_lambda_python_alpha import PythonFunction
import aws_cdk.aws_stepfunctions as sfn
import aws_cdk.aws_stepfunctions_tasks as sfn_tasks
from constructs import Construct

# Local Dependencies:
from ...shared import abs_path

TPS_ACQUIRER_LAMBDA_PATH = abs_path("fn-acquire-lock", __file__)


class SFnSemaphoreCleanUpChain(Construct):
    """Pseudo-chain to clean up an unused SFnSemaphore for a particular execution ID

    Expects inputs with properties:

    - $.ExecutionArn: (str) The ARN of the failed SFn execution to clean up locks for
    - $.LockName: (str) The unique name of the lock to be checked and cleaned
    - $.PerItemConcurrency: (int) The concurrency count reserved/cleared per execution

    Not quite an IChainable because one of the possible end states is a Choice, but you can next().
    """

    def __init__(
        self,
        scope: Construct,
        id: str,
        ddb_lock_table: dynamodb.Table,
        lock_id_attr: str,
        **kwargs,
    ):
        super().__init__(scope, id, **kwargs)

        self._start_state = sfn_tasks.DynamoGetItem(
            scope,
            "GetCurrentLockRecord",
            comment=(
                "Get info from DDB for the lock item to look and see if this specific owner is "
                "still holding a lock"
            ),
            result_path="$.lockinfo.currentlockitem",
            result_selector={
                "Item.$": "$.Item",
                "ItemString.$": "States.JsonToString($.Item)",
            },
            expression_attribute_names={
                "#lockownerid": sfn.JsonPath.string_at("$.ExecutionArn"),
            },
            key={
                lock_id_attr: sfn_tasks.DynamoAttributeValue.from_string(
                    sfn.JsonPath.string_at("$.LockName"),
                ),
            },
            projection_expression=[
                sfn_tasks.DynamoProjectionExpression().with_attribute("#lockownerid"),
            ],
            table=ddb_lock_table,
        )
        self._start_state.add_retry(
            backoff_rate=1.4,
            errors=["States.ALL"],
            interval=Duration.seconds(5),
            max_attempts=20,
        )

        self.check_lock_held_state = sfn.Choice(
            scope,
            "CheckIfLockIsHeld",
            comment=(
                "Check to see if the execution in question holds a lock, by looking for a 'Z' "
                "which indicates an ISO timestamp value in the stringified lock contents."
            ),
        )
        self._start_state.next(self.check_lock_held_state)

        self.clean_up_lock_state = sfn_tasks.DynamoUpdateItem(
            scope,
            "CleanUpLock",
            comment="If this lockowerid is still there, then clean it up and release the lock",
            return_values=sfn_tasks.DynamoReturnValues.UPDATED_NEW,
            condition_expression="attribute_exists(#lockownerid)",
            expression_attribute_names={
                "#currentlockcount": "currentlockcount",
                "#lockownerid": sfn.JsonPath.string_at("$.ExecutionArn"),
            },
            expression_attribute_values={
                ":decrease": sfn_tasks.DynamoAttributeValue.number_from_string(
                    sfn.JsonPath.string_at("States.JsonToString($.PerItemConcurrency)"),
                ),
            },
            key={
                lock_id_attr: sfn_tasks.DynamoAttributeValue.from_string(
                    sfn.JsonPath.string_at("$.LockName"),
                ),
            },
            table=ddb_lock_table,
            update_expression=(
                "SET #currentlockcount = #currentlockcount - :decrease REMOVE #lockownerid"
            ),
        )
        self.clean_up_lock_state.add_retry(
            errors=["DynamoDB.ConditionalCheckFailedException"],
            max_attempts=0,
        )
        self.clean_up_lock_state.add_retry(
            backoff_rate=1.8,
            errors=["States.ALL"],
            interval=Duration.seconds(5),
            max_attempts=15,
        )

        self.check_lock_held_state.when(
            sfn.Condition.and_(
                sfn.Condition.is_present("$.lockinfo.currentlockitem.ItemString"),
                sfn.Condition.string_matches("$.lockinfo.currentlockitem.ItemString", "*Z*"),
            ),
            self.clean_up_lock_state,
        )  # No otherwise - see next() fn

        self.fail_state = sfn.Fail(
            scope,
            "SemaphoreCleanUpFailed",
            comment="",
            error="SemaphoreCleanUpFailed",
            cause="Unable to clean up unused semaphore",
        )

        self.clean_up_lock_state.add_catch(
            self.fail_state,
            errors=["DynamoDB.ConditionalCheckFailedException"],
            result_path=None,
        )

        self._end_states = [self.clean_up_lock_state, self.check_lock_held_state]

    def next(self, next_step: sfn.IChainable) -> sfn.Chain:
        """Add next_step on to this pseudo-chain and return the sfn.Chain result"""
        self.check_lock_held_state.otherwise(next_step)
        return sfn.Chain.custom(
            self.start_state, [next_step], self.clean_up_lock_state.next(next_step)
        )

    @property
    def start_state(self):
        """The entry state for this pseudo-chain"""
        return self._start_state

    @property
    def end_states(self):
        """The exit states for this pseudo-chain"""
        return self._end_states


class SFnSemaphoreReaper(Construct):
    """A SFn state machine to clean up leaked locks from failed executions in SFnSemaphores

    One reaper may serve multiple state machines / SFnSemaphores. See .attach()
    """

    def __init__(
        self,
        scope: Construct,
        id: str,
        ddb_lock_table: dynamodb.Table,
        lock_id_attr: str,
        **kwargs,
    ):
        super().__init__(scope, id, **kwargs)
        self.attachments = []
        self.cleanup_chain = SFnSemaphoreCleanUpChain(
            self,
            "SemaphoreCleanUpChain",
            ddb_lock_table=ddb_lock_table,
            lock_id_attr=lock_id_attr,
        )
        success = sfn.Succeed(
            self,
            "SemaphoreCleanUpSucceeded",
            comment="Successfully checked and cleaned unused semaphores",
        )

        self.state_machine = sfn.StateMachine(
            self,
            "SemaphoreCleanUpMachine",
            definition=self.cleanup_chain.next(success),
            state_machine_type=sfn.StateMachineType.STANDARD,
            timeout=Duration.minutes(20),
        )

    def attach(
        self, state_machine: sfn.StateMachine, lock_name: str, per_item_concurrency: int = 1
    ):
        trigger_rule = aws_events.Rule(
            self,
            "SemaphoreCleanUpTrigger",
            description=(
                "Rule to trigger reaper for unused concurrency locks on failed SFn executions"
            ),
            enabled=True,
            event_pattern=aws_events.EventPattern(
                detail={
                    "stateMachineArn": [state_machine.state_machine_arn],
                    "status": ["ABORTED", "FAILED", "TIMED_OUT"],
                },
                source=["aws.states"],
            ),
            targets=[
                events_targets.SfnStateMachine(
                    self.state_machine,
                    input=aws_events.RuleTargetInput.from_object(
                        {
                            "LockName": lock_name,
                            "ExecutionArn": aws_events.EventField.from_path(
                                "$.detail.executionArn"
                            ),
                            "PerItemConcurrency": per_item_concurrency,
                        }
                    ),
                ),
            ],
        )
        self.state_machine.grant_start_execution(
            ServicePrincipal(
                service="events.amazonaws.com",
                conditions={"ArnEquals": {"aws:SourceArn": trigger_rule.rule_arn}},
            ),
        )
        self.attachments.append(trigger_rule)
        return self


class SFnSemaphoreDynamoDbTable(dynamodb.Table):
    """Optional DDB subclass to automatically configure default settings for SFnSemaphore tables"""

    def __init__(
        self,
        scope: Construct,
        id: str,
        lock_id_attr: str,
        billing_mode: dynamodb.BillingMode = dynamodb.BillingMode.PAY_PER_REQUEST,
        removal_policy: RemovalPolicy = RemovalPolicy.DESTROY,
        **kwargs,
    ):
        super().__init__(
            scope,
            id,
            partition_key=dynamodb.Attribute(
                name=lock_id_attr,
                type=dynamodb.AttributeType.STRING,
            ),
            billing_mode=billing_mode,
            removal_policy=removal_policy,
            **kwargs,
        )
        self._lock_id_attr = lock_id_attr

    @property
    def lock_id_attr(self) -> str:
        return self._lock_id_attr


class SFnSemaphoreLockAcquirer(Construct):
    """Pseudo-chain to acquire a lock for SFnSemaphore

    Uses a custom Lambda function if warmup_tps_limit is configured, else pure Step Functions.
    """

    def __init__(
        self,
        scope: Construct,
        id: str,
        ddb_lock_table: dynamodb.Table,
        lock_id_attr: str,
        lock_name: str,
        concurrency_limit: int,
        per_item_concurrency: int = 1,
        warmup_tps_limit: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(scope, id, **kwargs)

        if warmup_tps_limit:
            # To apply a TPS limit we need to do some timestamp processing so require a Lambda
            # function rather than just a Step Functions DynamoDB state for acquiring the lock:
            self.acq_lambda = PythonFunction(
                self,
                "AcquireLockFunction",
                entry=TPS_ACQUIRER_LAMBDA_PATH,
                index="main.py",
                handler="handler",
                memory_size=128,
                runtime=LambdaRuntime.PYTHON_3_9,
                timeout=Duration.seconds(60),
            )
            # (Read is required as well as write, for the item existence check when using Lambda)
            ddb_lock_table.grant_read_write_data(self.acq_lambda)
            self.acquire_lock_state = sfn_tasks.LambdaInvoke(
                self,
                "AcquireLockStep",
                comment="Attempt to acquire a lock with a conditional DynamoDB update operation",
                result_path=sfn.JsonPath.DISCARD,
                lambda_function=self.acq_lambda,
                payload=sfn.TaskInput.from_object(
                    {
                        "ConcurrencyLimit": concurrency_limit,
                        "ExecutionId": sfn.JsonPath.string_at("$$.Execution.Id"),
                        "LockIdAttribute": lock_id_attr,
                        "LockName": lock_name,
                        "PerItemConcurrency": per_item_concurrency,
                        "StateEnteredTime": sfn.JsonPath.string_at("$$.State.EnteredTime"),
                        "TableName": ddb_lock_table.table_name,
                        "WarmupTpsLimit": warmup_tps_limit,
                    }
                ),
                payload_response_only=True,
            )
        else:
            self.acquire_lock_state = sfn_tasks.DynamoUpdateItem(
                self,
                "AcquireLockStep",
                comment="Attempt to acquire a lock with a conditional DynamoDB update operation",
                result_path=sfn.JsonPath.DISCARD,
                key={lock_id_attr: sfn_tasks.DynamoAttributeValue.from_string(lock_name)},
                # Can't use < check because it fails with DynamoDB.ConditionalCheckFailedException
                # instead of DynamoDB.AmazonDynamoDBException when the lock item does not exist -
                # causing a States.Runtime error in the GetCurrentLockRecord state due to current
                # projection.
                # condition_expression="currentlockcount < :limit and attribute_not_exists(#lockownerid)",
                condition_expression="currentlockcount <> :limit and attribute_not_exists(#lockownerid)",
                expression_attribute_names={
                    "#currentlockcount": "currentlockcount",
                    "#lockownerid": sfn.JsonPath.string_at("$$.Execution.Id"),
                },
                expression_attribute_values={
                    ":increase": sfn_tasks.DynamoAttributeValue.from_number(per_item_concurrency),
                    ":limit": sfn_tasks.DynamoAttributeValue.from_number(concurrency_limit),
                    ":lockacquiredtime": sfn_tasks.DynamoAttributeValue.from_string(
                        sfn.JsonPath.string_at("$$.State.EnteredTime"),
                    ),
                },
                return_values=sfn_tasks.DynamoReturnValues.UPDATED_NEW,
                table=ddb_lock_table,
                update_expression=" ".join(
                    (
                        "SET",
                        "#currentlockcount = #currentlockcount + :increase,",
                        "#lockownerid = :lockacquiredtime",
                    )
                ),
            )
        self.acquire_lock_state.add_retry(
            errors=["DynamoDB.AmazonDynamoDBException", "Lambda_DynamoDBResourceNotFound"],
            max_attempts=0,
        )
        self.acquire_lock_state.add_retry(
            backoff_rate=2.0,
            errors=["States.ALL"],
            interval=Duration.seconds(2),
            max_attempts=6,
        )

        self.init_lock_state = sfn_tasks.DynamoPutItem(
            self,
            "InitializeLockStore",
            comment="Init the lock item in DynamoDB (for first case where it doesn't exist yet)",
            result_path=sfn.JsonPath.DISCARD,
            condition_expression=f"{lock_id_attr} <> :lockname",
            expression_attribute_values={
                ":lockname": sfn_tasks.DynamoAttributeValue.from_string(lock_name),
            },
            item={
                lock_id_attr: sfn_tasks.DynamoAttributeValue.from_string(lock_name),
                "currentlockcount": sfn_tasks.DynamoAttributeValue.from_number(0),
                "nextavailtime": sfn_tasks.DynamoAttributeValue.from_number(0),  # (Jan 1st 1970)
            },
            table=ddb_lock_table,
        )
        self.acquire_lock_state.add_catch(
            errors=["DynamoDB.AmazonDynamoDBException", "Lambda_DynamoDBResourceNotFound"],
            handler=self.init_lock_state,
            result_path=sfn.JsonPath.string_at("$.lockinfo.acquisitionerror"),
        )
        self.init_lock_state.add_catch(
            errors=["States.ALL"],
            handler=self.acquire_lock_state,
            result_path=sfn.JsonPath.DISCARD,
        )
        self.init_lock_state.next(self.acquire_lock_state)

        self.get_current_record_state = sfn_tasks.DynamoGetItem(
            self,
            "GetCurrentLockRecord",
            comment=(
                "When retry limit is exceeded or the execution may already hold a lock, load the "
                "current lock information from DDB so we can check if it's already held."
            ),
            result_path="$.lockinfo.currentlockitem",
            result_selector={
                # Using JsonToString because otherwise we have to check dynamically for the
                # existence of `$.Item.{$$.Execution.Id}` which I'm not aware of a way to do.
                "Item.$": "$.Item",
                "ItemString.$": "States.JsonToString($.Item)",
            },
            expression_attribute_names={
                "#lockownerid": sfn.JsonPath.string_at("$$.Execution.Id"),
            },
            key={lock_id_attr: sfn_tasks.DynamoAttributeValue.from_string(lock_name)},
            projection_expression=[
                sfn_tasks.DynamoProjectionExpression().with_attribute("#lockownerid"),
            ],
            table=ddb_lock_table,
        )
        self.acquire_lock_state.add_catch(
            errors=[
                "DynamoDB.ConditionalCheckFailedException",
                "Lambda_ConditionalCheckFailedException",
            ],
            handler=self.get_current_record_state,
            result_path=sfn.JsonPath.string_at("$.lockinfo.acquisitionerror"),
        )

        self.check_existing_lock_state = sfn.Choice(
            self,
            "CheckLockAlreadyAcquired",
            comment=(
                "Check lock information loaded from DDB to verify whether the current execution "
                "has already been granted a lock; or has not and should wait a while before "
                "retrying."
            ),
        )
        self.get_current_record_state.next(self.check_existing_lock_state)

        self.lock_already_acquired_state = sfn.Pass(
            self,
            "ContinueLockAlreadyAcquired",
            comment=(
                "In this state, we have confimed that lock is already held, so we pass the "
                "original execution input into the the function that does the work."
            ),
        )

        self.wait_to_get_lock_state = sfn.Wait(
            self,
            "WaitToGetLock",
            comment=(
                "If the lock indeed not been succesfully Acquired, then wait a while before "
                "trying again (in case of rate limits, etc)."
            ),
            time=sfn.WaitTime.duration(Duration.seconds(5)),
        )
        self.wait_to_get_lock_state.next(self.acquire_lock_state)
        if warmup_tps_limit:
            # If we're using the Lambda (which can detect throttling) then route throttles directly
            # to the waiter because native SFn GetItem seems to raise them as generic DDB error. If
            # DDB is overwhelmed/scaling, best is just to give it some time.
            self.acquire_lock_state.add_catch(
                errors=["Lambda_DynamoDBThrottlingException"],
                handler=self.wait_to_get_lock_state,
            )

        self.check_existing_lock_state.when(
            sfn.Condition.and_(
                sfn.Condition.is_present("$.lockinfo.currentlockitem.ItemString"),
                sfn.Condition.string_matches(
                    "$.lockinfo.currentlockitem.ItemString",
                    "*Z*",
                ),
            ),
            self.lock_already_acquired_state,
        ).otherwise(self.wait_to_get_lock_state)

        self._start_state = self.acquire_lock_state
        self._end_states = []

    @property
    def start_state(self):
        return self._start_state

    @property
    def end_states(self):
        return self._end_states


class SfnSemaphoreLockReleaseState(sfn_tasks.DynamoUpdateItem):
    """Step Functions state template for releasing the current SFnSemaphore lock"""

    def __init__(
        self,
        scope: Construct,
        id: str,
        table: dynamodb.ITable,
        lock_id_attr: str,
        lock_name: str,
        per_item_concurrency: int = 1,
        **kwargs,
    ):
        super().__init__(
            scope,
            id,
            result_path=sfn.JsonPath.DISCARD,
            condition_expression="attribute_exists(#lockownerid)",
            key={
                lock_id_attr: sfn_tasks.DynamoAttributeValue.from_string(lock_name),
            },
            expression_attribute_names={
                "#currentlockcount": "currentlockcount",
                "#lockownerid": sfn.JsonPath.string_at("$$.Execution.Id"),
            },
            expression_attribute_values={
                ":decrease": sfn_tasks.DynamoAttributeValue.from_number(per_item_concurrency),
            },
            return_values=sfn_tasks.DynamoReturnValues.UPDATED_NEW,
            table=table,
            update_expression=(
                "SET #currentlockcount = #currentlockcount - :decrease REMOVE #lockownerid"
            ),
            **kwargs,
        )
        self.add_retry(
            errors=["DynamoDB.ConditionalCheckFailedException"],
            max_attempts=0,
        )
        self.add_retry(
            errors=["States.ALL"],
            backoff_rate=1.5,
            max_attempts=5,
        )


class SFnSemaphore(Construct):
    """Construct using a DynamoDB semaphore pattern to limit concurrency in Step Functions graphs

    Wraps a chain of work states to initially acquire (or wait for) a concurrency lock, and then
    release the lock after completion.

    Use this semaphore in your state machine definition via the `.chain` property, and remember to
    attach a corresponding `SFnSemaphoreReaper` to catch any leaked locks due to execution
    failures. You can also chain this semaphore's `.release_state` elsewhere in your graph if
    needed.
    """

    def __init__(
        self,
        scope: Construct,
        id: str,
        workchain: sfn.IChainable,
        ddb_lock_table: dynamodb.Table,
        lock_id_attr: str,
        lock_name: str,
        concurrency_limit: int,
        per_item_concurrency: int = 1,
        warmup_tps_limit: Optional[float] = None,
        **kwargs,
    ):
        """Create a SFnSemaphore

        Arguments
        ---------
        scope : Construct
            CDK construct scope
        id : str
            CDK construct ID
        workchain : sfn.IChainable
            The `Chain` of Step Functions states to wrap in the concurrency-limiting semaphore
        ddb_lock_table : dynamodb.Table
            The DynamoDB table in which concurrency locks will be tracked for this semaphore
        lock_id_attr : str
            The attribute name where lock IDs are stored in the given `ddb_lock_table`
        lock_name : str
            The name for this semaphore lock, unique within the scope of `ddb_lock_table`
        concurrency_limit : int
            The maximum concurrency to enforce for this work chain
        per_item_concurrency : int
            The per-execution concurrency for this workchain: Default 1
        warmup_tps_limit : Optional[float]
            Optional limit for time between lock grants, in grants per second.
        **kwargs :
            Passed through to parent Construct class
        """
        super().__init__(scope, id, **kwargs)

        self.lock_acquirer = SFnSemaphoreLockAcquirer(
            self,
            "LockAcquisitionChain",
            ddb_lock_table=ddb_lock_table,
            lock_id_attr=lock_id_attr,
            lock_name=lock_name,
            concurrency_limit=concurrency_limit,
            per_item_concurrency=per_item_concurrency,
            warmup_tps_limit=warmup_tps_limit,
        )

        self._start_state = sfn.Parallel(
            self,
            "GetLock",
            comment=(
                "This parallel state contains the logic to acquire a lock and to handle the cases "
                "where a lock cannot be Acquired. Containing this in a parallel allows for visual "
                "separation when viewing the state machine and makes it easier to reuse this same "
                "logic elsewhere if desired. Because this state sets ResultPath: null, it will "
                "not manipulate the execution input that is passed on to the subsequent part of "
                "your statemachine that is responsible for doing the work."
            ),
            result_path=sfn.JsonPath.DISCARD,
        ).branch(self.lock_acquirer.start_state)

        self._release_state = SfnSemaphoreLockReleaseState(
            self,
            "ReleaseLock",
            comment="Release the semaphore lock after the process is finished",
            table=ddb_lock_table,
            lock_id_attr=lock_id_attr,
            lock_name=lock_name,
            per_item_concurrency=per_item_concurrency,
        )

        self.workchain = workchain
        self._chain = self.start_state.next(workchain).next(self._release_state)
        self._end_states = [self._release_state]

    @property
    def chain(self):
        """The overall state chain wrapping the provided `workchain` in a semaphore"""
        return self._chain

    @property
    def start_state(self):
        """The entry state for this SFn graph"""
        return self._start_state

    @property
    def end_states(self):
        """The end states for this SFn graph are simply [release_state]"""
        return self._end_states

    @property
    def release_state(self):
        """The state which releases the semaphore lock for this construct"""
        return self._release_state
