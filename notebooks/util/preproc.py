# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Data pre-processing utilities
"""

# Python Built-Ins:
from __future__ import annotations
from collections import deque
from datetime import datetime
import json
from logging import getLogger
from math import ceil
import os
import re
import time
from typing import Callable, Iterable, List, Optional, Tuple, Union

# External Dependencies:
import boto3
import numpy as np
from tqdm.notebook import tqdm  # Progress bars
import trp  # Amazon Textract Response Parser

# Local Dependencies:
from . import uid


logger = getLogger("preproc")
s3 = boto3.resource("s3")
s3client = boto3.client("s3")
sfn = boto3.client("stepfunctions")


def s3uri_to_bucket_and_key(s3uri: str) -> Tuple[str, str]:
    """Convert an s3://... URI string to a (bucket, key) tuple"""
    if not s3uri.lower().startswith("s3://"):
        raise ValueError(f"Expected S3 object URI to start with s3://. Got: {s3uri}")
    bucket, _, key = s3uri[len("s3://") :].partition("/")
    return bucket, key


def s3_object_exists(bucket_name_or_s3uri: str, key: Optional[str] = None) -> bool:
    """Check if an object exists in Amazon S3

    Parameters
    ----------
    bucket_name_or_s3uri :
        Either an 's3://.../...' object URI, or an S3 bucket name.
    key :
        Ignored if `bucket_name_or_s3uri` is a full URI, otherwise mandatory: Key of the object to
        check.
    """
    if bucket_name_or_s3uri.lower().startswith("s3://"):
        bucket_name, key = s3uri_to_bucket_and_key(bucket_name_or_s3uri)
    elif not key:
        raise ValueError(
            "key is mandatory when bucket_name_or_s3uri is not an s3:// URI. Got: %s"
            % bucket_name_or_s3uri
        )
    else:
        bucket_name = bucket_name_or_s3uri
    try:
        s3client.head_object(Bucket=bucket_name, Key=key)
        return True
    except s3client.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        else:
            raise e


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
    skip_existing=True,
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
                            pstart.update(1)
                            pbar.update(1)
                            continue

                        # Else req_or_existing is a SFn request:
                        request = req_or_existing
                    except Exception as err:
                        pbar.write(f"Exception preparing Textract parameters for: {item}")
                        results[ix][manifest_out_field] = err
                        pstart.update(1)
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
                    pstart.update(1)

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


class DataManifestWarning:
    """Descriptor object for a warning/issue in generating a data manifest"""

    def __init__(
        self,
        textract_s3uri: str,
        img_candidates: List[str],
        n_textract_pages: int,
        doc_s3uri: Optional[str] = None,
    ):
        """Create a DataManifestWarning

        Parameters
        ----------
        textract_s3uri : str
            's3://...' URI of the Textract result for the document
        img_candidates : List[str]
            List of S3 keys found in search for page images
        n_textract_pages : int
            Expected number of pages in doc per the Textract result
        rec_doc_s3uri : str
            's3://...' URI of the raw document if present
        """
        self.textract_s3uri = textract_s3uri
        self.img_candidates = img_candidates
        self.n_textract_pages = n_textract_pages
        self.doc_s3uri = doc_s3uri


def trp_page_has_content(page: trp.Page) -> bool:
    return len(page.lines) > 0


def find_cleaned_page_imgs_by_rel_file_path(
    rel_filepath: str,
    imgs_s3uri: str,
) -> Tuple[List[str], List[Union[int, None]]]:
    """Find cleaned page images (and their expected page numbers) on S3 for a doc in the corpus

    This function essentially reconstructs logic applied by the image cleaning pre-processing job
    to locate cleaned images in S3 for a given raw document in the corpus: Including multi-page
    PDFs, TIFFs, or single-page input images like JPEGs. Returned objects are verified to actually
    exist in S3 at the time the function was called.

    Parameters
    ----------
    rel_filepath : str
        Relative path to a source document or image in the corpus (i.e. within the data/raw folder)
    imgs_s3uri : str
        's3://...' root URI under which cleaned page images are stored, with filenames generated
        from documents as per `clean_dataset_for_img_ocr()`

    Returns
    -------
    img_candidate_s3keys: List[str]
        List of S3 object keys which (have been tested to exist and) are expected to correspond to
        cleaned page images of the input document. Not necessarily in page number order.
    img_candidate_pagenums: List[Union[str, NoneType]]
        Inferred (1-based) page number for each entry in `img_candidate_s3keys`, or `None` if page
        number could not be inferred for that object.
    """
    # pdf2image outputs look like {MyOriginalFileBaseName}-0000-00.{FileExt}:
    PDF2IMAGE_REGEX = re.compile(r"^-\d{4,}-\d+.(?:png|jpg|jpeg)$", re.IGNORECASE)
    NONPDF_REGEX = re.compile(r"^(-\d{4,})?.(?:png|jpg|jpeg)$", re.IGNORECASE)

    imgs_bucket_name, _, imgs_s3key_root = imgs_s3uri[len("s3://") :].partition("/")
    imgs_bucket = s3.Bucket(imgs_bucket_name)

    rel_filedir, _, filename = rel_filepath.rpartition("/")
    filename_root, _, extension = filename.rpartition(".")
    extension = extension.lower()
    file_img_s3key_prefix = "".join(
        (
            imgs_s3key_root,
            "/",
            rel_filedir + "/" if rel_filedir else "",
            filename_root,
        )
    )

    raw_candidate_objs = imgs_bucket.objects.filter(Prefix=file_img_s3key_prefix)

    if extension == "pdf":
        # Use the pdf2image regex to find images and associate page numbers:
        img_candidate_s3keys = list(
            map(
                lambda o: o.key,
                filter(
                    lambda o: PDF2IMAGE_REGEX.match(o.key[len(file_img_s3key_prefix) :]),
                    raw_candidate_objs,
                ),
            )
        )
        img_candidate_pagenums = list(
            map(
                lambda f: int(f.rpartition(".")[0].rpartition("-")[2]),
                img_candidate_s3keys,
            )
        )
    else:
        # Could be a single-page (e.g. PNG) or multi-page (e.g. TIFF) image:
        raw_candidate_s3keys = [o.key for o in raw_candidate_objs]
        regex_matches = [
            NONPDF_REGEX.match(k[len(file_img_s3key_prefix) :]) for k in raw_candidate_s3keys
        ]

        img_candidate_s3keys = [
            raw_candidate_s3keys[ix] for ix in range(len(regex_matches)) if regex_matches[ix]
        ]

        if len(img_candidate_s3keys) == 1:
            img_candidate_pagenums = [1]
        else:
            img_candidate_pagenums = [int(match.group(1)) for match in regex_matches if match]

    return img_candidate_s3keys, img_candidate_pagenums


def collate_data_manifest(
    manifest_file: str,
    input_manifest: Union[str, Iterable[dict]],
    textract_s3_prefix: str,
    imgs_s3_prefix: str,
    raw_s3_prefix: Optional[str] = None,
    by: str = "page",
    no_content: Optional[str] = None,
    progress_desc: str = "Building data manifest...",
) -> List[DataManifestWarning]:
    """Build a data manifest with validations that the required artifacts actually exist on S3

    Writes a JSON-lines manifest file to the given path, with each record containing as standard:

    - For **page-based manifests**: 'page-num' (int, 1-based), 'source-ref' (str, image URI),
      'textract-ref' (str, Textract result URI)
    - For **doc-based manifests**: 'source-ref' (str, Textract result URI), 'page-refs' (List[str],
      page image URIs)

    Other props may also be included (see parameters below):

    Parameters
    ----------
    manifest_file :
        File name/path to output to
    input_manifest :
        Path to a JSONLines input manifest of objects (typically docs the page images linked yet),
        or an in-memory list/iterable of objects of the same.
    textract_s3_prefix :
        's3://...' root URI under which Textract results are stored, used for mapping from Textract
        result URIs to expected page image URIs.
    imgs_s3_prefix :
        's3://...' root URI under which cleaned page images are stored, with filenames generated
        from documents as per `clean_dataset_for_img_ocr()`
    by :
        Set 'page' (default) to produce one manifest record per page; or 'doc' to produce one
        manifest record per doc with an array of page images.
    no_content :
        Set 'omit' to skip pages with no text content detected by Textract (i.e. not generate a
        record for page-based manifest; omit from 'page-refs' for doc-based manifest). Set 'flag'
        to add a 'has-content' (bool, for page-based) or 'pages-have-content' (List[bool], for
        doc-based) attribute to the output. (Default None - no checking)
        manifest:
    progress_desc :
        Description label for the progress bar (Default 'Building data manifest...')

    Returns
    -------
    warnings :
        List of docs excluded from the manifest due to some inconsistency between Textract result
        and page images on S3. If len()==0, you're good to go. Otherwise, investigate.
    """
    # Tidy up some arguments:
    if textract_s3_prefix.endswith("/"):
        textract_s3_prefix = textract_s3_prefix[:-1]
    if not textract_s3_prefix.lower().startswith("s3://"):
        raise ValueError(
            f"textract_s3_prefix must be a valid s3://... URI. Got: {textract_s3_prefix}"
        )
    if imgs_s3_prefix.endswith("/"):
        imgs_s3_prefix = imgs_s3_prefix[:-1]
    if not imgs_s3_prefix.lower().startswith("s3://"):
        raise ValueError(f"imgs_s3_prefix must be a valid s3://... URI. Got: {imgs_s3_prefix}")
    if raw_s3_prefix:
        if raw_s3_prefix.endswith("/"):
            raw_s3_prefix = raw_s3_prefix[:-1]
        if not raw_s3_prefix.lower().startswith("s3://"):
            raise ValueError(
                f"raw_s3_prefix, if provided, must be a valid s3://... URI. Got: {raw_s3_prefix}"
            )
    by = by.lower()
    if by not in ("page", "doc"):
        raise ValueError(f"Manifest must be `by` either 'page' or 'doc'. Got: {by}")
    if no_content:
        no_content = no_content.lower()
        if no_content not in ("omit", "flag"):
            raise ValueError(
                f"`no_content` option must be 'omit', 'flag', or None. Got: {no_content}"
            )

    # If input objects provided by file path rather than in-memory, read them in:
    if isinstance(input_manifest, str):
        with open(input_manifest) as f:
            input_manifest = [json.loads(line) for line in f]

    imgs_bucket_name, _ = s3uri_to_bucket_and_key(imgs_s3_prefix)

    warnings: List[DataManifestWarning] = []
    with open(manifest_file, "w") as fmanifest:
        for item in tqdm(input_manifest, desc=progress_desc):
            # If raw file listed, check it exists:
            rec_doc_s3uri = item.get("raw-ref")
            if rec_doc_s3uri:
                if not s3_object_exists(rec_doc_s3uri):
                    raise ValueError(f"Raw document ('raw-ref') missing from S3: {rec_doc_s3uri}")

            # Load the consolidated Textract JSON:
            rec_tex_s3uri = item["textract-ref"]
            rec_tex_bucket, rec_tex_key = s3uri_to_bucket_and_key(rec_tex_s3uri)
            try:
                doc = trp.Document(
                    json.loads(s3.Object(rec_tex_bucket, rec_tex_key).get()["Body"].read())
                )
            except Exception as e:
                print(f"Failed to open Textract object {rec_tex_s3uri}")
                raise e
            if no_content:
                pages_have_content = [trp_page_has_content(p) for p in doc.pages]

            # Try to map page images from the raw doc URI and prefix if available:
            if raw_s3_prefix and rec_doc_s3uri and rec_doc_s3uri.startswith(raw_s3_prefix):
                mapped_from_raw = True
                doc_relpath = rec_doc_s3uri[len(raw_s3_prefix) + 1 :]
                (
                    img_candidate_s3keys,
                    img_candidate_pagenums,
                ) = find_cleaned_page_imgs_by_rel_file_path(
                    doc_relpath,
                    imgs_s3uri=imgs_s3_prefix,
                )
                if len(img_candidate_s3keys) == 0:
                    logger.warn(
                        "No page images found from raw doc path '%s'... Trying from textract-ref",
                        doc_relpath,
                    )
            else:
                mapped_from_raw = False
                img_candidate_s3keys = []
                img_candidate_pagenums = []

            if len(img_candidate_s3keys) == 0:
                if rec_tex_s3uri.startswith(textract_s3_prefix):
                    doc_relpath = rec_tex_s3uri[len(textract_s3_prefix) + 1 :]
                    if doc_relpath.endswith("/consolidated.json"):
                        doc_relpath = doc_relpath.rpartition("/")[0]
                    # List the matching page images in S3:
                    (
                        img_candidate_s3keys,
                        img_candidate_pagenums,
                    ) = find_cleaned_page_imgs_by_rel_file_path(
                        doc_relpath,
                        imgs_s3uri=imgs_s3_prefix,
                    )
                elif not mapped_from_raw:
                    # Couldn't map from either raw-ref or textract-ref
                    logger.warn(
                        "textract-ref did not start with textract_s3_prefix and could also not "
                        "map from raw-ref / raw_s3_prefix."
                    )

            if img_candidate_pagenums != list(range(1, len(doc.pages) + 1)):
                if len(img_candidate_pagenums) == 0:
                    logger.warn("No page images found for doc, excluding from manifest:\n%s", item)
                else:
                    logger.warn("Mismatch in doc, excluding from manifest:\n%s", item)
                warnings.append(
                    DataManifestWarning(
                        textract_s3uri=rec_tex_s3uri,
                        img_candidates=img_candidate_s3keys,
                        n_textract_pages=len(doc.pages),
                        doc_s3uri=rec_doc_s3uri,
                    )
                )
                continue

            # Write the manifest entry/entries:
            if by == "page":
                for page_ix in range(0, len(doc.pages)):
                    record = {
                        **item,
                        "source-ref": f"s3://{imgs_bucket_name}/{img_candidate_s3keys[page_ix]}",
                        "page-num": page_ix + 1,
                    }
                    if no_content == "omit":
                        if not pages_have_content[page_ix]:
                            continue
                    elif no_content == "flag":
                        record["has-content"] = pages_have_content[page_ix]
                    fmanifest.write(json.dumps(record) + "\n")
            else:
                record = {
                    **item,
                    "source-ref": rec_tex_s3uri,
                    "page-refs": list(
                        map(
                            lambda key: f"s3://{imgs_bucket_name}/{key}",
                            img_candidate_s3keys,
                        )
                    ),
                }
                if no_content == "omit":
                    record["page-refs"] = [
                        ixval[1]
                        for ixval in filter(
                            lambda ixval: pages_have_content[ixval[0]],
                            enumerate(record["page-refs"]),
                        )
                    ]
                elif no_content == "flag":
                    record["pages-have-content"] = pages_have_content
                fmanifest.write(json.dumps(record) + "\n")
    return warnings
