# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Data pre-processing utilities
"""

# Python Built-Ins:
from collections import deque
from datetime import datetime
import json
from logging import getLogger
from math import ceil
import re
import time
from types import SimpleNamespace
from typing import Iterable, List, Optional, Tuple, Union

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


def s3_object_exists(bucket_name: str, key: str) -> bool:
    try:
        s3client.head_object(Bucket=bucket_name, Key=key)
        return True
    except s3client.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        else:
            raise e


def call_textract(
    textract_sfn_arn: str,
    input_base_s3uri: Optional[str],
    input_relpaths: List[str],
    features: List[str] = ["FORMS", "TABLES"],
    output_base_s3uri: Optional[str] = None,
    skip_existing=True,
) -> List[Union[str, dict, Exception]]:
    """Process a list of documents in S3 with Textract via AWS Step Functions

    Uses the Textract-only Step Functions State Machine set up by this solution's CloudFormation
    stack to process a batch of documents in S3 - with concurrency and rate limit controls built
    in to the state machine. Also does a little client-side rate management to try and optimize
    costs.

    Arguments
    ---------
    textract_sfn_arn : str
        The ARN of the "plain Textract state machine" (i.e. Textract only, no post-processing
        pipeline) created by this stack. See util.project.init() for help fetching this via code.
    input_base_s3uri : str
        The s3://... URI under which your input documents are collected
    input_relpaths : List[str]
        The paths relative to (under) the input_base_s3uri) of each document you want to process.
    features : List[str]
        Which additional Textract features you'd like to call in addition to DetectText. Default
        ["FORMS", "TABLES"]
    output_base_s3uri : Optional[str]
        Must be an s3://... URI if set. Results will be saved in S3 by Textract, at
        `{output_base_s3uri}/{input_relpath}/consolidated.json`.
    skip_existing : bool
        Before processing each doc, check whether a consolidated output JSON already exists for it
        in S3 and skip if so. Only supported when consolidated S3 output is enabled. Default True.

    Returns
    -------
    results : List[Union[str, dict, Exception]]
        Results in same order as input input_relpaths. Results of type string are s3:// URI strings
        pointing to the output JSON. Other types represent errors: Typically results of a failed
        Step Functions DescribeExecution call or a caught Exception.
    """
    if not input_base_s3uri.lower().startswith("s3://"):
        raise ValueError(f"input_base_s3uri must be an S3 URI. Got {input_base_s3uri}")
    if not input_base_s3uri.endswith("/"):
        input_base_s3uri += "/"
    input_bucket, _, input_prefix = input_base_s3uri[len("s3://") :].partition("/")

    if output_base_s3uri is not None:
        if not output_base_s3uri.lower().startswith("s3://"):
            raise ValueError(
                f"output_base_s3uri, if specified, must be an S3 URI. Got {output_base_s3uri}"
            )
        if not output_base_s3uri.endswith("/"):
            output_base_s3uri += "/"
        output_bucket, _, output_prefix = output_base_s3uri[len("s3://") :].partition("/")
    else:
        output_bucket = None
        output_prefix = None

    results = [None] * len(input_relpaths)
    # Looping first through to create the jobs and then to check their status would be simple, but
    # would obscure initial insight on progress for big batches and would also present the most
    # challenging/costly possible burst profile for the state machine to manage. By instead
    # building a combined event loop where we push new jobs and check some earlier ones, we can
    # improve on both aspects.
    enumd_queue_relpaths = deque(enumerate(input_relpaths))
    sfn_submissions = []
    with tqdm(desc="Textracting PDFs...", total=len(input_relpaths)) as pbar:
        with tqdm(desc="Starting jobs...", total=len(input_relpaths)) as pstart:
            # Limit speed of new job creation to be snappy initially, but then cool off to more
            # sustainable pace when many jobs have been started:
            jobs_started = 0
            t_next_startable = datetime.now().timestamp()
            t_start_checking_results = t_next_startable + 16  # Min expected process latency secs

            def inter_start_wait_secs(jobs_started: int) -> float:
                """Linear scaling from X=.2s to Y=1s over N=100 jobs after first T=20 jobs"""
                return 0.2 + (1.0 - 0.2) * min(100, max(0, jobs_started - 20)) / 100

            # Main event loop:
            while len(enumd_queue_relpaths) or len(sfn_submissions):
                # INPUT STEP: Try to process one queued path:
                if len(enumd_queue_relpaths):
                    ix, rel_filepath = enumd_queue_relpaths.popleft()
                    input_key = f"{input_prefix}{rel_filepath}"
                    consolidated_key = f"{output_prefix}{rel_filepath}/consolidated.json"
                    consolidated_s3uri = f"s3://{output_bucket}/{consolidated_key}"
                    if skip_existing:
                        if s3_object_exists(output_bucket, consolidated_key):
                            pbar.write(f"Skipping already-textracted '{rel_filepath}'")
                            results[ix] = consolidated_s3uri
                            pstart.update(1)
                            pbar.update(1)
                            continue

                    request = {
                        "Input": {
                            "Bucket": input_bucket,
                            "Key": input_key,
                        },
                        "Output": {
                            "Features": features,
                            "Key": consolidated_key,
                        },
                    }
                    if output_bucket:
                        request["Output"]["Bucket"] = output_bucket

                    now = datetime.now().timestamp()
                    if t_next_startable > now:
                        time.sleep(t_next_startable - now)
                    t_next_startable = datetime.now().timestamp() + inter_start_wait_secs(
                        jobs_started
                    )

                    try:
                        sfn_resp = sfn.start_execution(
                            stateMachineArn=textract_sfn_arn,
                            name=uid.append_timestamp("notebook"),
                            input=json.dumps(request),
                        )
                        jobs_started += 1
                        sfn_submissions.append(
                            SimpleNamespace(
                                input_doc=f"s3://{input_bucket}/{input_key}",
                                ix=ix,
                                execution_arn=sfn_resp["executionArn"],
                                started=sfn_resp["startDate"],
                            )
                        )
                    except Exception as err:
                        pbar.write(f"Exception creating Textract job for: {input_key}")
                        results[ix] = err
                        pbar.update(1)
                    pstart.update(1)

                # OUTPUT STEP: Try to query (some or all, depending on phase) results
                input_items_queued = len(enumd_queue_relpaths) > 0
                if (datetime.now().timestamp() >= t_start_checking_results) and len(
                    sfn_submissions
                ):

                    def check_submission(submission):
                        """Return submission if still processing, else None if done. Update results"""
                        desc = sfn.describe_execution(executionArn=submission.execution_arn)
                        status = desc["status"]
                        result = None
                        if status == "SUCCEEDED":
                            sfn_output = json.loads(desc.get("output", "{}"))
                            result_s3uri = sfn_output.get("Output", {}).get("S3Uri")
                            if result_s3uri:
                                results[submission.ix] = result_s3uri
                            else:
                                pbar.write(
                                    f"Doc processed but missing output S3 URI: "
                                    f"{submission.input_doc}"
                                )
                                results[submission.ix] = desc
                            pbar.update(1)
                        elif status != "RUNNING":
                            results[submission.ix] = desc
                            pbar.write(
                                "Doc failed to process - see results for details: "
                                f"{submission.input_doc}"
                            )
                            pbar.update(1)
                        else:
                            result = submission
                        time.sleep(0.08)
                        return result

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
                        for ix_sub in sample_ixs:
                            sfn_submissions[ix_sub] = check_submission(sfn_submissions[ix_sub])
                        sfn_submissions = list(filter(lambda s: s, sfn_submissions))
                    else:
                        # When all jobs are submitted, this is the only active section of the event
                        # loop: Check the whole list and pause a while between loops.
                        sfn_submissions = list(
                            filter(lambda s: s, map(check_submission, sfn_submissions))
                        )
                        time.sleep(3)
    return results


class DataManifestWarning:
    """Descriptor object for a warning/issue in generating a data manifest"""

    def __init__(
        self,
        textract_s3uri: str,
        rel_filepath: str,
        img_candidates: List[str],
        n_textract_pages: int,
    ):
        """Create a DataManifestWarning

        Parameters
        ----------
        textract_s3uri : str
            's3://...' URI of the Textract result for the document
        rel_filepath : str
            Relative file path of the document being processed
        img_candidates : List[str]
            List of S3 keys found in search for page images
        n_textract_pages : int
            Expected number of pages in doc per the Textract result
        """
        self.textract_s3uri = textract_s3uri
        self.rel_filepath = rel_filepath
        self.img_candidates = img_candidates
        self.n_textract_pages = n_textract_pages


def trp_page_has_content(page: trp.Page) -> bool:
    return len(page.lines) > 0


def find_cleaned_page_imgs_by_textract_uri(
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
        Relative path to a raw document or image in the corpus (i.e. within the data/raw folder)
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


def build_data_manifest(
    manifest_file: str,
    rel_doc_paths: Iterable[str],
    textract_s3uri: str,
    imgs_s3uri: str,
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
    manifest_file : str
        File name/path to output to
    rel_doc_paths : Iterable[str]
        Relative paths to each document to be included in the manifest
    textract_s3uri : str
        's3://...' root URI under which Textract results are stored
    imgs_s3uri : str
        's3://...' root URI under which cleaned page images are stored, with filenames generated
        from documents as per `clean_dataset_for_img_ocr()`
    by : str
        Set 'page' (default) to produce one manifest record per page; or 'doc' to produce one
        manifest record per doc with an array of page images.
    no_content : Optional[str]
        Set 'omit' to skip pages with no text content detected by Textract (i.e. not generate a
        record for page-based manifest; omit from 'page-refs' for doc-based manifest). Set 'flag'
        to add a 'has-content' (bool, for page-based) or 'pages-have-content' (List[bool], for
        doc-based) attribute to the output. (Default None - no checking)
        manifest:
    progress_desc : str
        Description label for the progress bar (Default 'Building data manifest...')

    Returns
    -------
    warnings : List[DataManifestWarning]
        List of docs excluded from the manifest due to some inconsistency between Textract result
        and page images on S3. If len()==0, you're good to go. Otherwise, investigate.
    """
    # Tidy up some arguments:
    if textract_s3uri.endswith("/"):
        textract_s3uri = textract_s3uri[:-1]
    if not textract_s3uri.lower().startswith("s3://"):
        raise ValueError(f"textract_s3uri must be a valid s3://... URI. Got: {textract_s3uri}")
    if imgs_s3uri.endswith("/"):
        imgs_s3uri = imgs_s3uri[:-1]
    if not imgs_s3uri.lower().startswith("s3://"):
        raise ValueError(f"imgs_s3uri must be a valid s3://... URI. Got: {imgs_s3uri}")
    by = by.lower()
    if by not in ("page", "doc"):
        raise ValueError(f"Manifest must be `by` either 'page' or 'doc'. Got: {by}")
    if no_content:
        no_content = no_content.lower()
        if no_content not in ("omit", "flag"):
            raise ValueError(
                f"`no_content` option must be 'omit', 'flag', or None. Got: {no_content}"
            )

    imgs_bucket_name = imgs_s3uri[len("s3://") :].partition("/")[0]
    textract_bucket_name, _, textract_s3key_root = textract_s3uri[len("s3://") :].partition("/")
    textract_bucket = s3.Bucket(textract_bucket_name)

    warnings = []
    with open(manifest_file, "w") as fmanifest:
        for rel_filepath in tqdm(rel_doc_paths, desc=progress_desc):
            # Load the consolidated Textract JSON:
            file_textract_s3key = f"{textract_s3key_root}/{rel_filepath}/consolidated.json"
            file_textract_s3uri = f"s3://{textract_bucket_name}/{file_textract_s3key}"
            try:
                doc = trp.Document(
                    json.loads(textract_bucket.Object(file_textract_s3key).get()["Body"].read())
                )
            except Exception as e:
                print(f"Failed to open Textract object {file_textract_s3uri}")
                raise e
            if no_content:
                pages_have_content = [trp_page_has_content(p) for p in doc.pages]

            # List the matching page images in S3:
            img_candidate_s3keys, img_candidate_pagenums = find_cleaned_page_imgs_by_textract_uri(
                rel_filepath,
                imgs_s3uri=imgs_s3uri,
            )
            if img_candidate_pagenums != list(range(1, len(doc.pages) + 1)):
                if len(img_candidate_pagenums) == 0:
                    logger.warn(
                        f"No page images found for doc, excluding from manifest: {rel_filepath}"
                    )
                else:
                    logger.warn(f"Mismatch in doc, excluding from manifest: {rel_filepath}")
                warnings.append(
                    DataManifestWarning(
                        textract_s3uri=file_textract_s3uri,
                        rel_filepath=rel_filepath,
                        img_candidates=img_candidate_s3keys,
                        n_textract_pages=len(doc.pages),
                    )
                )
                continue

            # Write the manifest entry/entries:
            if by == "page":
                for page_ix in range(0, len(doc.pages)):
                    record = {
                        "source-ref": f"s3://{imgs_bucket_name}/{img_candidate_s3keys[page_ix]}",
                        "textract-ref": file_textract_s3uri,
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
                    "source-ref": file_textract_s3uri,
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
