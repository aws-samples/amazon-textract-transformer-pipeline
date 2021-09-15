# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Customized DDB Semaphore lock acquisition function with timestamp processing

While vanilla concurrency locking can be done in Step Functions only, enforcing an additional rate
limit on the speed with which new locks are granted requires some extra timestamp processing which
is not possible in SFn alone (where timestamps are strings). Inspired by the below AWS blog &
source code sample:

https://aws.amazon.com/blogs/compute/controlling-concurrency-in-distributed-systems-using-aws-step-functions/

https://github.com/aws-samples/aws-stepfunctions-examples/tree/main/sam/app-control-concurrency-with-dynamodb
"""

# Python Built-Ins:
from datetime import datetime
from decimal import Decimal
import logging
import os

# External Dependencies:
import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ddbclient = boto3.client("dynamodb")
ddb = boto3.resource("dynamodb")

# This function is able to consume environment variable defaults, but it's probably preferable to
# pass these parameters in via the Step Function so you have the option to share one Lambda between
# multiple SFn semaphores:
DEFAULT_DDB_TABLE_NAME = os.environ.get("DEFAULT_DDB_TABLE_NAME")
DEFAULT_LOCK_ID_ATTRIBUTE = os.environ.get("DEFAULT_LOCK_ID_ATTRIBUTE")
DEFAULT_LOCK_NAME = os.environ.get("DEFAULT_LOCK_NAME")
DEFAULT_PER_ITEM_CONCURRENCY = int(os.environ.get("DEFAULT_PER_ITEM_CONCURRENCY", "1"))
DEFAULT_CONCURRENCY_LIMIT = int(os.environ.get("DEFAULT_CONCURRENCY_LIMIT", "0"))
DEFAULT_WARMUP_TPS_LIMIT = float(os.environ.get("DEFAULT_WARMUP_TPS_LIMIT", "0"))


class MalformedRequest(ValueError):
    """Named Step Functions error class for missing/malformed input properties"""

    pass


class Lambda_OtherDynamoDBError(ValueError):
    """Named Step Functions error class for generic DynamoDB exception"""

    pass


class Lambda_ConditionalCheckFailedException(ValueError):
    """Named Step Functions error class for DynamoDB conditional update failure"""

    pass


class Lambda_DynamoDBResourceNotFound(ValueError):
    """Named Step Functions error class for semaphore lock object missing in DynamoDB"""

    pass


class Lambda_DynamoDBThrottlingException(ValueError):
    """Named Step Functions error class for DynamoDB throttling (after retries)"""

    pass


class AcquireLockEvent:
    """Parser for retrieving event fields from Lambda input data and/or environment variables"""

    # Required event attrs:
    execution_id: str
    state_entered_time: str

    # With environment variable defaults:
    concurrency_limit: int
    ddb_table_name: str
    lock_id_attr: str
    lock_name: str
    per_item_concurrency: int
    warmup_tps_limit: float

    def __init__(self, raw_event: dict):
        # Required event attrs first:
        try:
            self.execution_id = raw_event["ExecutionId"]
            self.state_entered_time = raw_event["StateEnteredTime"]
        except KeyError as kerr:
            raise MalformedRequest(f"Input event missing required key '{kerr}'") from kerr

        # Params with environment variable defaults:
        self.ddb_table_name = raw_event.get("TableName", DEFAULT_DDB_TABLE_NAME)
        if not self.ddb_table_name:
            raise MalformedRequest(
                "Must provide either input key 'TableName' or env var DEFAULT_DDB_TABLE_NAME"
            )
        self.lock_id_attr = raw_event.get("LockIdAttribute", DEFAULT_LOCK_ID_ATTRIBUTE)
        if not self.lock_id_attr:
            raise MalformedRequest(
                "Must provide either input key 'LockIdAttribute' or env var "
                "DEFAULT_LOCK_ID_ATTRIBUTE"
            )
        self.lock_name = raw_event.get("LockName", DEFAULT_LOCK_NAME)
        if not self.lock_name:
            raise MalformedRequest(
                "Must provide either input key 'LockName' or env var DEFAULT_LOCK_NAME"
            )
        self.per_item_concurrency = raw_event.get(
            "PerItemConcurrency",
            DEFAULT_PER_ITEM_CONCURRENCY,
        )
        if not (self.per_item_concurrency and self.per_item_concurrency > 0):
            raise MalformedRequest(
                "Input key 'PerItemConcurrency' (or env var DEFAULT_PER_ITEM_CONCURRENCY) must be "
                f"an integer greater than 0. Got '{self.per_item_concurrency}"
            )
        self.concurrency_limit = raw_event.get("ConcurrencyLimit", DEFAULT_CONCURRENCY_LIMIT)
        self.warmup_tps_limit = raw_event.get("WarmupTpsLimit", DEFAULT_WARMUP_TPS_LIMIT)


def handler(event, context):
    logger.info("Received event: %s", event)
    event = AcquireLockEvent(event)

    # time.time() is only guaranteed second precision, we'd like ms or better for multi-TPS:
    current_timestamp = datetime.now().timestamp()
    ddb_table = ddb.Table(event.ddb_table_name)
    ddb_key = {event.lock_id_attr: event.lock_name}
    try:
        resp = ddb_table.update_item(
            Key=ddb_key,
            ReturnValues="UPDATED_NEW",
            ConditionExpression=" and ".join(
                (
                    "currentlockcount < :limit",
                    "attribute_not_exists(#lockownerid)",
                    "nextavailtime < :currenttime",
                )
            ),
            ExpressionAttributeNames={
                "#currentlockcount": "currentlockcount",
                "#lockownerid": event.execution_id,
                "#nextavailtime": "nextavailtime",
            },
            ExpressionAttributeValues={
                ":increase": event.per_item_concurrency,
                ":limit": event.concurrency_limit,
                ":currenttime": Decimal(current_timestamp),
                ":nextavailtime": Decimal(
                    current_timestamp + (1.0 / event.warmup_tps_limit)
                    if event.warmup_tps_limit
                    else 0
                ),
                ":lockacquiredtime": event.state_entered_time,
            },
            UpdateExpression=" ".join(
                (
                    "SET",
                    "#currentlockcount = #currentlockcount + :increase,"
                    "#lockownerid = :lockacquiredtime,",
                    "#nextavailtime = :nextavailtime",
                )
            ),
        )
    except ddbclient.exceptions.ConditionalCheckFailedException as err_conditional:
        logger.error("Conditional check failed")
        # Because of the nature of the checks enforced by this acquirer (which yields this error
        # even if the lock item does not yet exist), and the fact that we can't yet catch
        # non-existent lock items in the next state because they trigger States.Runtime errors;
        # we'll perform a quick getItem check here and raise a different error if the item is
        # missing:
        try:
            get_resp = ddb_table.get_item(
                Key=ddb_key,
                ProjectionExpression=next(k for k in ddb_key),  # Just the (first) key attribute
                ReturnConsumedCapacity="NONE",
            )
        except Exception:
            logger.exception("Additional exception while checking for lock item existence")
            get_resp = {"Error": "Couldn't check item exists"}

        if not (get_resp and get_resp.get("Item")):
            raise Lambda_DynamoDBResourceNotFound("Lock item does not exist") from err_conditional
        else:
            raise Lambda_ConditionalCheckFailedException(str(err_conditional)) from err_conditional
    except (
        ddbclient.exceptions.ProvisionedThroughputExceededException,
        ddbclient.exceptions.RequestLimitExceeded,
        ddbclient.exceptions.ThrottlingException,
    ) as err_throttle:
        logger.error("DynamoDB throttled")
        raise Lambda_DynamoDBThrottlingException(str(err_throttle)) from err_throttle
    except Exception as err_general:
        logger.error("Other DynamoDB error")
        raise Lambda_OtherDynamoDBError(str(err_general)) from err_general

    logger.info("DDB Response: %s", resp)
    # resp["Attributes"] is a pre-parsed dict direct from attr name to value, including the updated
    # keys. Since some of these are Decimals, Decimals can't be JSON dumped, and the output of this
    # Lambda step is discarded anyway, we'll just return an empty object.
    return {}
