# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Notebook-simplifying utilities for OCR with Amazon Textract
"""
# Python Built-Ins:
from __future__ import annotations
from collections import deque
from datetime import datetime
import json
from logging import getLogger
from math import ceil
import os
import time
from typing import Callable, List, Optional, Union

# External Dependencies:
import boto3
import numpy as np
from tqdm.notebook import tqdm  # Progress bars

# Local Dependencies:
from . import uid
from .s3 import s3_object_exists, s3uri_to_bucket_and_key


logger = getLogger("ocr")
smclient = boto3.client("sagemaker")
sfn = boto3.client("stepfunctions")
tagging = boto3.client("resourcegroupstaggingapi")


def list_preannotated_textract_uris(
    ann_jobs_folder: str = "data/annotations",
    exclude_job_names: List[str] = [],
) -> List[str]:
    """List out (alphabetically) the textract-ref URIs for which annotations already exist locally

    Parameters
    ----------
    ann_jobs_folder :
        A local folder containing the outputs of multiple SageMaker Ground Truth jobs (either
        directly output manifest files, or the folder sub-structures created by SMGT in S3).
    exclude_job_names :
        List of any job names under `ann_jobs_folder` to exclude from the check.
    """
    uris = set()  # Protect against introducing duplicates
    for job in os.listdir(ann_jobs_folder):
        if job in exclude_job_names:
            logger.info("Skipping excluded job %s", job)
            continue

        job_path = os.path.join(ann_jobs_folder, job)
        if os.path.isfile(job_path):
            manifest_file = job_path
        else:
            manifest_file = os.path.join(job_path, "manifests", "output", "output.manifest")
            if not os.path.isfile(manifest_file):
                logger.warning("Skipping job %s: No output manifest at %s", job_path, manifest_file)
                continue

        try:
            with open(manifest_file) as f:
                uris.update([json.loads(line)["textract-ref"] for line in f])
        except json.JSONDecodeError:
            logger.warning("%s file is not valid JSON-Lines: Skipping", manifest_file)

    return sorted(uris)


class TextractSFnSubmission:
    """Class to create and track OCR requests to Textract-only Step Functions state machine

    For batch processing, use the `call_textract()` function, which uses this class internally but
    manages concurrency and warm-up. Otherwise, general usage flow of this class is to call
    `.prepare_request()`, `.create()`, then `.check_result()`.
    """

    def __init__(self, doc_s3uri: str, execution_arn: str, started: str, ix: Optional[int] = None):
        """Create a TextractSfnSubmission from already-known state

        If you're using this class to submit requests, you want .create() instead!

        Parameters
        ----------
        doc_s3uri :
            s3://... URI of input document
        execution_arn :
            Execution ARN from AWS Step Functions
        started :
            ISO Date/time SFn execution was started (e.g. `startDate` field from SFn response)
        ix :
            Optional index number to track this submission in a list
        """
        self.doc_s3uri = doc_s3uri
        self.execution_arn = execution_arn
        self.started = started
        self.ix = ix

    @staticmethod
    def inter_start_wait_secs(jobs_started: int) -> float:
        """Calculate the recommended waiting period between starting Textract SFn jobs

        Linear scaling from X=.2s to Y=1s over N=100 jobs after first T=20 jobs
        """
        return 0.2 + (1.0 - 0.2) * min(100, max(0, jobs_started - 20)) / 100

    @staticmethod
    def prepare_request(
        manifest_item: dict,
        manifest_raw_field: str = "raw-ref",
        manifest_out_field: str = "textract-ref",
        output_base_s3uri: Optional[str] = None,
        input_base_s3uri: Optional[str] = None,
        features: List[str] = ["FORMS", "TABLES"],
        skip_existing: bool = True,
    ) -> Union[dict, str]:
        """Prepare a SFn request payload from a manifest line item (or return pre-existing)

        Includes logic to set sensible defaults on S3 output location, if an output URI is not
        explicitly set by the manifest.

        Parameters
        ----------
        manifest_item :
            Loaded JSON dict from a manifest, detailing this (single) file to be processed.
        manifest_raw_field :
            Property name on the `manifest_item` at which the S3 URI of the input (raw) document to
            be Textracted is given. Default "raw-ref".
        manifest_out_field :
            Property name on the `manifest_item` at which the S3 URI of the output (Textract result)
            will be saved. If the input record already has this field, it will override other output
            location settings (allowing you to override output location per-document).
        output_base_s3uri :
            Must be an s3://... URI if set (no trailing slash). If provided, results will be saved
            under this S3 path unless otherwise specified per-document by the input manifest (see
            `manifest_out_field`). If only this param is used, results will be saved to:
            `{output_base_s3uri}/{entire_input_key}/consolidated.json`. If this parameter is not
            set, output will be saved directly to `/consolidated.json` under the source document's
            S3 URI.
        input_base_s3uri :
            Must be an s3://... URI if set (no trailing slash). If set alongside
            `output_base_s3uri`, `input_base_s3uri` will be stripped off the beginning of source doc
            URIs before mapping the remaining relative path to the `output_base_s3uri` location.
            I.e. results will be saved to:
            `{output_base_s3uri}/{input_rel_to_input_base_s3uri}/consolidated.json`. Ignored if
            `output_base_s3uri` is not set.
        features :
            Which additional Textract features you'd like to call in addition to DetectText. Default
            ["FORMS", "TABLES"]
        skip_existing :
            Before processing each doc, check whether a consolidated output JSON already exists for
            it in S3 and skip if so. Only supported when consolidated S3 output is enabled. Default
            True.

        Returns
        -------
        request_or_s3ri :
            Either a `dict` request payload for AWS Step Functions / this class' `.create()` method,
            or an "s3://..." URI if `skip_existing` is enabled and the Textract result already
            exists.
        """
        if input_base_s3uri:
            if not input_base_s3uri.lower().startswith("s3://"):
                raise ValueError(
                    f"input_base_s3uri, if specified, must be an S3 URI. Got: {input_base_s3uri}"
                )
            if not input_base_s3uri.endswith("/"):
                input_base_s3uri += "/"
            input_bucket, _ = s3uri_to_bucket_and_key(input_base_s3uri)
        else:
            input_bucket = None

        if output_base_s3uri:
            if not output_base_s3uri.lower().startswith("s3://"):
                raise ValueError(
                    f"output_base_s3uri, if specified, must be an S3 URI. Got {output_base_s3uri}"
                )
            if not output_base_s3uri.endswith("/"):
                output_base_s3uri += "/"
            output_bucket, output_prefix = s3uri_to_bucket_and_key(output_base_s3uri)
        else:
            output_bucket = None
            output_prefix = None

        doc_s3uri = manifest_item[manifest_raw_field]
        doc_bucket, doc_key = s3uri_to_bucket_and_key(doc_s3uri)
        prev_out_s3uri = manifest_item.get(manifest_out_field)
        if prev_out_s3uri:
            out_s3uri = prev_out_s3uri
            out_bucket, out_key = s3uri_to_bucket_and_key(out_s3uri)
        elif output_base_s3uri:
            out_bucket = output_bucket
            if input_base_s3uri:
                if not doc_s3uri.startswith(input_base_s3uri):
                    raise ValueError(
                        "Input document URI '%s' does not start with provided "
                        "input_base_s3uri '%s'" % (doc_s3uri, input_base_s3uri)
                    )
                relpath = doc_s3uri[len(input_base_s3uri) :]
            else:
                relpath = doc_key
            out_key = "".join((output_prefix, relpath, "/consolidated.json"))
        else:
            out_bucket = input_bucket
            out_key = doc_key + "/consolidated.json"

        if skip_existing:
            if s3_object_exists(out_bucket, out_key):
                return f"s3://{out_bucket}/{out_key}"

        # Format expected by the Textract-only SFn State Machine:
        return {
            "Input": {
                "Bucket": doc_bucket,
                "Key": doc_key,
            },
            "Output": {
                "Features": features,
                "Key": out_key,
                "Bucket": out_bucket,
            },
        }

    @classmethod
    def create(
        cls,
        state_machine_arn: str,
        request: dict,
        ix: Optional[int] = None,
    ) -> TextractSFnSubmission:
        """Submit a job to Textract-only State Machine and create a Submission tracker object

        Parameters
        ----------
        state_machine_arn :
            ARN of the Textract-only pipeline state machine
        request :
            Step Functions payload object (see `.prepare_request()`)
        ix :
            Optional index number to track this submission in a list
        """
        doc_s3uri = "".join(
            (
                "s3://",
                request["Input"]["Bucket"],
                "/",
                request["Input"]["Key"],
            )
        )
        sfn_resp = sfn.start_execution(
            stateMachineArn=state_machine_arn,
            name=uid.append_timestamp("notebook"),
            input=json.dumps(request),
        )
        return cls(
            doc_s3uri=doc_s3uri,
            execution_arn=sfn_resp["executionArn"],
            started=sfn_resp["startDate"],
            ix=ix,
        )

    def check_result(
        self,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> Union[str, dict, None]:
        """Check the status of this submission, returning result if complete

        Parameters
        ----------
        log_fn :
            Optional callable function to log diagnostics. No logs will be generated if `None`.

        Returns
        -------
        result :
            If the submission is still processing, returns None. If completed successfully, returns
            the 's3://...' URI string pointing to the result JSON. If the job failed or the result
            URI could not be found, returns states:DescribeExecution result.
        """
        if log_fn is None:
            log_fn = lambda _: None
        desc = sfn.describe_execution(executionArn=self.execution_arn)
        status = desc["status"]
        if status == "SUCCEEDED":
            sfn_output = json.loads(desc.get("output", "{}"))
            result_s3uri = sfn_output.get("Output", {}).get("S3Uri")
            if result_s3uri:
                return result_s3uri
            else:
                log_fn(f"Doc processed but missing output S3 URI: {self.doc_s3uri}")
                return desc
        elif status != "RUNNING":
            log_fn(f"Doc failed to process - see results for details: {self.doc_s3uri}")
            return desc
        else:
            return None


def call_textract(
    textract_sfn_arn: str,
    input_manifest: Union[str, List[dict]],
    manifest_raw_field: str = "raw-ref",
    manifest_out_field: str = "textract-ref",
    output_base_s3uri: Optional[str] = None,
    input_base_s3uri: Optional[str] = None,
    features: List[str] = ["FORMS", "TABLES"],
    skip_existing: bool = True,
) -> List[dict]:
    """Process a set of documents in S3 with Textract via AWS Step Functions

    Uses the Textract-only Step Functions State Machine set up by this solution's CloudFormation
    stack to process a batch of documents in S3 - with concurrency and rate limit controls built
    in to the state machine. Also does a little client-side rate management to try and optimize
    costs and speed.

    Parameters
    ----------
    textract_sfn_arn :
        The ARN of the "plain Textract state machine" (i.e. Textract only, no post-processing
        pipeline) created by this stack. See util.project.init() for help fetching this via code.
    input_manifest :
        Path to a JSONLines manifest of objects, or an in-memory list/iterable of objects detailing
        the files to be processed.
    manifest_raw_field :
        Property name on the input manifest at which the S3 URI of the input (raw) document to be
        Textracted is given. Default "raw-ref".
    manifest_out_field :
        Property name on the manifest at which the S3 URI of the output (Textract result) will be
        saved. If input records already have this field, it will override other output location
        settings (allowing you to override output location per-document).
    output_base_s3uri :
        Must be an s3://... URI if set (no trailing slash). If provided, results will be saved
        under this S3 path unless otherwise specified per-document by the input manifest (see
        `manifest_out_field`). If only this param is used, results will be saved to:
        `{output_base_s3uri}/{entire_input_key}/consolidated.json`. If this parameter is not set,
        output will be saved directly to `/consolidated.json` under the source document's S3 URI.
    input_base_s3uri :
        Must be an s3://... URI if set (no trailing slash). If set alongside `output_base_s3uri`,
        `input_base_s3uri` will be stripped off the beginning of source doc URIs before mapping the
        remaining relative path to the `output_base_s3uri` location. I.e. results will be saved to:
        `{output_base_s3uri}/{input_rel_to_input_base_s3uri}/consolidated.json`. Ignored if
        `output_base_s3uri` is not set.
    features :
        Which additional Textract features you'd like to call in addition to DetectText. Default
        ["FORMS", "TABLES"]
    skip_existing :
        Before processing each doc, check whether a consolidated output JSON already exists for it
        in S3 and skip if so. Only supported when consolidated S3 output is enabled. Default True.

    Returns
    -------
    results :
        Output manifest objects in same order as input (shallow copies), with `manifest_out_field`
        set one of 3 ways: An s3:// URI string, where the Textraction was successful; an Exception,
        where some error was logged; or a dict Step Functions DescribeExecution call giving more
        details if some other failure occurred.
    """
    if isinstance(input_manifest, str):
        with open(input_manifest) as f:
            input_manifest = [json.loads(line) for line in f]

    # Output manifest is a shallow copy of input objects, so we can override the output field:
    results = [{**item} for item in input_manifest]

    # Looping first through to create the jobs and then to check their status would be simple, but
    # would obscure initial insight on progress for big batches and would also present the most
    # challenging/costly possible burst profile for the state machine to manage. By instead
    # building a combined event loop where we push new jobs and check some earlier ones, we can
    # improve on both aspects.
    enumd_queue_items = deque(enumerate(input_manifest))
    sfn_submissions: List[TextractSFnSubmission] = []
    with tqdm(desc="Textracting PDFs...", total=len(input_manifest)) as pbar:
        with tqdm(desc="Starting jobs...", total=len(input_manifest)) as pstart:
            # Limit speed of new job creation to be snappy initially, but then cool off to more
            # sustainable pace when many jobs have been started:
            jobs_started = 0
            t_next_startable = datetime.now().timestamp()
            t_start_checking_results = t_next_startable + 16  # Min expected process latency secs

            def increment_pstart():
                """Increment job start progress *and* close the progress bar if completed

                We want to close pstart as soon as all jobs are started, in order to display the
                correct (shorter) total runtime rather than waiting until jobs are also finished.
                """
                pstart.update(1)
                if pstart.n >= pstart.total:
                    pstart.close()

            # Main event loop:
            while len(enumd_queue_items) or len(sfn_submissions):
                # INPUT STEP: Try to process one queued item:
                if len(enumd_queue_items):
                    ix, item = enumd_queue_items.popleft()
                    doc_s3uri = item[manifest_raw_field]
                    try:
                        req_or_existing = TextractSFnSubmission.prepare_request(
                            item,
                            manifest_raw_field=manifest_raw_field,
                            manifest_out_field=manifest_out_field,
                            output_base_s3uri=output_base_s3uri,
                            input_base_s3uri=input_base_s3uri,
                            features=features,
                            skip_existing=skip_existing,
                        )

                        if isinstance(req_or_existing, str):
                            pbar.write(f"Skipping already-textracted '{doc_s3uri}'")
                            results[ix][manifest_out_field] = req_or_existing
                            increment_pstart()
                            pbar.update(1)
                            continue

                        # Else req_or_existing is a SFn request:
                        request = req_or_existing
                    except Exception as err:
                        pbar.write(f"Exception preparing Textract parameters for: {item}")
                        results[ix][manifest_out_field] = err
                        increment_pstart()
                        pbar.update(1)
                        continue

                    now = datetime.now().timestamp()
                    if t_next_startable > now:
                        time.sleep(t_next_startable - now)
                    t_next_startable = (
                        datetime.now().timestamp()
                        + TextractSFnSubmission.inter_start_wait_secs(jobs_started)
                    )

                    try:
                        sfn_submissions.append(
                            TextractSFnSubmission.create(
                                state_machine_arn=textract_sfn_arn,
                                request=request,
                                ix=ix,
                            )
                        )
                        jobs_started += 1
                    except Exception as err:
                        pbar.write(f"Exception creating Textract job for: {doc_s3uri}")
                        results[ix][manifest_out_field] = err
                        pbar.update(1)
                    increment_pstart()

                # OUTPUT STEP: Try to query (some or all, depending on phase) results
                input_items_queued = len(enumd_queue_items) > 0
                if (datetime.now().timestamp() >= t_start_checking_results) and len(
                    sfn_submissions
                ):
                    if input_items_queued:
                        # When still creating jobs, our goal is to give an approximate view of
                        # completion progress without slowing down job creation too much. We'll
                        # randomly sample {N/40} of the earliest {1/2} of submissions to test:
                        # Restricting to early subset because we know older executions are more
                        # likely to have completed, but randomizing because in-order processing
                        # is not guaranteed - so taking a deterministic approach would (badly)
                        # under-estimate number of completed jobs.
                        #
                        # This length-scaled sampling approach has the added benefit of slowing
                        # our loop down if the number of active jobs becomes >>{40}
                        sample_ixs = np.random.choice(
                            ceil(len(sfn_submissions) / 4),
                            ceil(len(sfn_submissions) / 40),
                            replace=False,
                        )
                    else:
                        # When all jobs are submitted, this is the only active section of the event
                        # loop: Check the whole list and also pause a while between loops.
                        sample_ixs = list(range(len(sfn_submissions)))
                        time.sleep(3)

                    for ix_sub in sample_ixs:
                        sub = sfn_submissions[ix_sub]
                        result = sub.check_result(log_fn=pbar.write)
                        time.sleep(0.08)
                        if result is not None:
                            results[sub.ix][manifest_out_field] = result
                            sfn_submissions[ix_sub] = None
                            pbar.update(1)
                    sfn_submissions = list(filter(lambda s: s, sfn_submissions))
    return results


def describe_sagemaker_ocr_model(engine_name: str) -> dict:
    """Look up a SageMaker OCR model created by the solution CDK stack by OCR engine name

    Returns
    -------
    result :
        Result of calling SageMaker DescribeModel API on the located model

    Raises
    ------
    ValueError
        If no matching models could be found in SageMaker
    """
    candidate_arns = [
        obj["ResourceARN"]
        for obj in tagging.get_resources(
            TagFilters=[
                {
                    "Key": "OCREngineName",
                    "Values": [engine_name],
                },
            ],
            ResourceTypeFilters=["sagemaker:model"],
        )["ResourceTagMappingList"]
    ]
    n_candidates = len(candidate_arns)

    if n_candidates < 1:
        raise ValueError(
            "Couldn't find any SageMaker models with tag 'OCREngineName' = '%s'. Check your CDK "
            "solution deployed with BUILD_SM_OCRS including this engine, or locate your intended "
            "OCR model by hand." % (engine_name,)
        )
    elif n_candidates > 1:
        logger.warning(
            "Found %s SageMaker models with tag 'OCREngineName' = '%s'. Taking first in list: %s",
            n_candidates,
            engine_name,
            n_candidates,
        )

    model_name = candidate_arns[0].partition("/")[2]
    return smclient.describe_model(ModelName=model_name)
