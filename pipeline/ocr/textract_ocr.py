# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CDK for core OCR stage of the document processing pipeline with Amazon Textract
"""
# External Dependencies:
from aws_cdk import Duration, RemovalPolicy
import aws_cdk.aws_dynamodb as dynamodb
from aws_cdk.aws_iam import Effect, PolicyDocument, PolicyStatement, Role, ServicePrincipal
from aws_cdk.aws_lambda import Runtime as LambdaRuntime
from aws_cdk.aws_lambda_python_alpha import PythonFunction
from aws_cdk.aws_s3 import Bucket
import aws_cdk.aws_sns as sns
import aws_cdk.aws_sns_subscriptions as subs
import aws_cdk.aws_stepfunctions as sfn
import aws_cdk.aws_stepfunctions_tasks as sfn_tasks
from constructs import Construct

# Local Dependencies:
from . import sfn_semaphore
from ..shared import abs_path


TEXTRACT_LAMBDA_PATH = abs_path("fn-call-textract", __file__)


class TextractOCRStep(Construct):
    """CDK Construct for a concurrency- & TPS-limiting Amazon Textract OCR step in a Step Function

    This construct's `.sfn_task` expects inputs with $.Input.Bucket and $.Input.Key properties
    specifying the location of the raw input document, and will return an object with Bucket and
    Key pointing to the consolidated Textract JSON object.
    """

    def __init__(
        self,
        scope: Construct,
        id: str,
        lambda_role: Role,
        output_bucket: Bucket,
        output_prefix: str,
        concurrency_limit: int = 90,
        warmup_tps_limit: float = 2,
        lambda_memory_size: int = 1024,
        lambda_timeout: Duration = Duration.minutes(3),
        timeout_excluding_queue: Duration = Duration.minutes(25),
        timeout_including_queue: Duration = Duration.minutes(30),
        **kwargs,
    ):
        """Create a TextractOCRStep

        Parameters
        ----------
        scope :
            CDK construct scope
        id :
            CDK construct ID
        lambda_role :
            IAM Role that the Textract-invoking Lambda function will run with
        output_bucket :
            (Pre-existing) S3 bucket where Textract result files should be stored
        output_prefix :
            Prefix under which Textract result files should be stored in S3 (under this prefix,
            the original input document keys will be mapped).
        concurrency_limit :
            Maximum number of Textract jobs which may be in-progress at a time. Additional requests
            will be pooled for retry via AWS Step Functions (order not guaranteed). Refer to your
            account & region's Amazon Textract quotas to set, and consider reducing a margin in
            case your usage is limited more in practice by the rate of result-fetching Get*** APIs
            than the total job concurrency limit.
        warmup_tps_limit :
            Limit on maximum rate of new Textract job creation. Additional requests will be pooled
            for retry via AWS Step Functions (order not guaranteed). Refer to your account &
            region's Amazon Textract Quotas.
        lambda_memory_size :
            MB of memory to reserve for the Lambda function that calls Amazon Textract and
            consolidates response objects. You may need to increase this setting if dealing with
            very large documents, to ensure the full consolidated Textract response JSON fits in
            memory.
        lambda_timeout :
            Time-out for the Lambda function that calls Amazon Textract and consolidates response
            objects. You may need to increase this setting if dealing with documents so large that
            consolidation takes a long time, or allowing the Lambda to poll/retry Textract API calls
            over a longer period.
        timeout_excluding_queue :
            Timeout for the Textract processing job itself to be considered as failed.
        timeout_including_queue :
            Timeout for the end-to-end step (including concurrency management / queuing) to be
            considered as failed.
        """
        super().__init__(scope, id, **kwargs)

        lambda_role.add_to_policy(
            PolicyStatement(
                sid="CallTextract",
                actions=[
                    "textract:AnalyzeDocument",
                    "textract:DetectDocumentText",
                    "textract:GetDocumentAnalysis",
                    "textract:GetDocumentTextDetection",
                    "textract:StartDocumentAnalysis",
                    "textract:StartDocumentTextDetection",
                ],
                effect=Effect.ALLOW,
                resources=["*"],
            )
        )
        self.ddb_table = dynamodb.Table(
            self,
            "TextractAsyncStateCacheTable",
            partition_key=dynamodb.Attribute(
                name="TextractJobId",
                type=dynamodb.AttributeType.STRING,
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
            time_to_live_attribute="ExpiresAt",
        )
        lambda_role.add_to_policy(
            PolicyStatement(
                sid="DDBAsyncStateCacheTableAccess",
                actions=["dynamodb:GetItem", "dynamodb:PutItem"],
                effect=Effect.ALLOW,
                resources=[self.ddb_table.table_arn],
            )
        )
        self.sns_topic = sns.Topic(self, "TextractCallbackTopic")
        self.textract_sns_role = Role(
            self,
            "TextractSNSRole",
            description="Execution role to give Textract permission to notify our SNS topic",
            assumed_by=ServicePrincipal("textract.amazonaws.com"),
            inline_policies={
                "SNSAccess": PolicyDocument(
                    statements=[
                        PolicyStatement(
                            sid="SNSTopicAccess",
                            actions=["sns:Publish"],
                            effect=Effect.ALLOW,
                            resources=[self.sns_topic.topic_arn],
                        ),
                    ],
                ),
            },
        )
        lambda_role.add_to_policy(
            PolicyStatement(
                sid="PassTextractSNSRole",
                actions=["iam:PassRole"],
                effect=Effect.ALLOW,
                resources=[self.textract_sns_role.role_arn],
            )
        )
        self.caller_lambda = PythonFunction(
            self,
            "CallTextract",
            entry=TEXTRACT_LAMBDA_PATH,
            environment={
                "CALLBACK_SNS_ROLE_ARN": self.textract_sns_role.role_arn,
                "CALLBACK_SNS_TOPIC_ARN": self.sns_topic.topic_arn,
                "DDB_STATE_CACHE_TABLE": self.ddb_table.table_name,
            },
            index="main.py",
            handler="handler",
            memory_size=lambda_memory_size,
            role=lambda_role,
            runtime=LambdaRuntime.PYTHON_3_9,
            timeout=lambda_timeout,
        )
        self.sns_topic.add_subscription(subs.LambdaSubscription(self.caller_lambda))

        self.inner_textact_step = sfn_tasks.LambdaInvoke(
            self,
            "TextractLambdaStep",
            comment="Extract document text and structure with Amazon Textract",
            lambda_function=self.caller_lambda,
            payload=sfn.TaskInput.from_object(
                {
                    "IdempotencySalt": sfn.JsonPath.task_token,
                    "Input": sfn.JsonPath.string_at("$.Input"),
                    "Output": sfn.JsonPath.string_at("$.Output"),
                    "TaskToken": sfn.JsonPath.task_token,
                }
            ),
            result_path="$.Output",  # Overwrite 'Output'
            integration_pattern=sfn.IntegrationPattern.WAIT_FOR_TASK_TOKEN,
            timeout=timeout_excluding_queue,
        )

        # We'll use a Step Function semaphore construct to manage concurrency and TPS for Textract,
        # from the sfn_semaphore folder. While in general it might be nice to share the semaphore
        # DB table and reaper machine between semaphores, this CDK app only uses this one lock.
        # Therefore we'll just init everything here and not over-complicate this construct's API:
        global_lock_name = "TextractConcurrencyLock"  # Could consider: cdk.Names.unique_id(self)
        self.semaphore_ddb = sfn_semaphore.SFnSemaphoreDynamoDbTable(
            self,
            "TextractSemaphoreDDB",
            lock_id_attr="LockName",
        )
        self.semaphore_reaper = sfn_semaphore.SFnSemaphoreReaper(
            self,
            "TextractSemaphoreReaper",
            ddb_lock_table=self.semaphore_ddb,
            lock_id_attr=self.semaphore_ddb.lock_id_attr,
        )
        self.semaphore = sfn_semaphore.SFnSemaphore(
            self,
            "TextractSemaphore",
            workchain=self.inner_textact_step,
            ddb_lock_table=self.semaphore_ddb,
            lock_id_attr=self.semaphore_ddb.lock_id_attr,
            lock_name=global_lock_name,
            concurrency_limit=concurrency_limit,
            warmup_tps_limit=warmup_tps_limit,
        )
        self.textract_state_machine = sfn.StateMachine(
            self,
            "TextractStateMachine",
            definition=self.semaphore.chain,
            timeout=timeout_including_queue,
        )
        self.semaphore_reaper.attach(self.textract_state_machine, lock_name=global_lock_name)

        # This inner, semaphore-wrapped state machine will be the task as seen by the OCR pipeline:
        self.sfn_task = sfn_tasks.StepFunctionsStartExecution(
            self,
            "TextractTask",
            comment="Extract document text and structure with Amazon Textract",
            state_machine=self.textract_state_machine,
            input=sfn.TaskInput.from_object(
                {
                    "Input": {
                        "Bucket": sfn.JsonPath.string_at("$.Input.Bucket"),
                        "Key": sfn.JsonPath.string_at("$.Input.Key"),
                    },
                    "Output": {
                        "Bucket": output_bucket.bucket_name,
                        "Prefix": output_prefix,
                    },
                }
            ),
            integration_pattern=sfn.IntegrationPattern.RUN_JOB,
            timeout=timeout_including_queue,
            # We overwrite the entire SFN state here because this step is used in a Parallel with
            # thumbnail generation:
            output_path=sfn.JsonPath.string_at("$.Output.Output"),
        )
