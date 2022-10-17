# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Data pre-processing and manifest consolidation utilities
"""
# Python Built-Ins:
from __future__ import annotations
import json
from logging import getLogger
import os
from random import Random
import re
from typing import Dict, Iterable, List, Optional, Set, TextIO, Tuple, Union

# External Dependencies:
import boto3
from sagemaker.estimator import Framework
from tqdm.notebook import tqdm  # Progress bars
import trp  # Amazon Textract Response Parser

# Local Dependencies:
from .s3 import s3_object_exists, s3uri_to_bucket_and_key, s3uri_to_relative_path


logger = getLogger("preproc")
s3 = boto3.resource("s3")


class DummyFramework(Framework):
    """A dummy to allow running FrameworkProcessor jobs on custom containers with unknown base FWs

    Use this class with `sagemaker.processing.FrameworkProcessor`, to enable running framework-mode
    (source_dir of multiple scripts, perhaps with a requirements.txt) processing jobs on customized
    DLCs without worrying about the exact framework and version parameters.

    FrameworkProcessor usually requires an `estimator_cls` parameter (e.g. PyTorch, HuggingFace) and
    specific framework version information - but this is mainly used for looking up standard
    container image URIs. Most of the relevant implementation for running processing is shared
    across frameworks.

    If you only have a single script file, use `ScriptProcessor` instead for simplicity and
    stability. If you know exactly what framework and version your container should run with, use
    the specific framework's processor (e.g. `PyTorchProcessor`). When you have a framework-based
    container image and want to run a bundle of Python files on it, agnostic of exactly what
    framework/version it is, you can use this as follows:

    ```python
    processor = FrameworkProcessor(
        estimator_cls=DummyFramework,
        image_uri=image_uri,  # Your custom, framework-based container
        framework_version="",  # Can be blank in this case
        py_version="",  # Can be blank in this case
        ...
    )
    ```
    """

    _framework_name = "dummy"

    def create_model(self, **kwargs):
        """Dummy implementation - raises NotImplementedError if called

        This method must be implemented to allow instantiating the abstract parent class, but it's
        not needed for our intended use (running framework-agnostic processing jobs).
        """
        raise NotImplementedError("DummyFramework does not implement 'create_model'")


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
                    logger.warning(
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
                    logger.warning(
                        "textract-ref did not start with textract_s3_prefix and could also not "
                        "map from raw-ref / raw_s3_prefix."
                    )

            if img_candidate_pagenums != list(range(1, len(doc.pages) + 1)):
                if len(img_candidate_pagenums) == 0:
                    logger.warning(
                        "No page images found for doc, excluding from manifest:\n%s",
                        item,
                    )
                else:
                    logger.warning("Mismatch in doc, excluding from manifest:\n%s", item)
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


def list_preannotated_img_paths(
    annotations_folder: str = "data/annotations",
    exclude_job_names: List[str] = [],
    key_prefix: str = "data/imgs-clean/",
) -> Set[str]:
    """Find the set of relative image paths that have already been annotated"""
    filepaths = set()  # Protect against introducing duplicates
    for job_folder in os.listdir(annotations_folder):
        if job_folder in exclude_job_names:
            logger.info(f"Skipping excluded job {job_folder}")
            continue
        manifest_file = os.path.join(
            "data",
            "annotations",
            job_folder,
            "manifests",
            "output",
            "output.manifest",
        )
        if not os.path.isfile(manifest_file):
            if os.path.isdir(os.path.join(annotations_folder, job_folder)):
                logger.warning(f"Skipping job {job_folder}: No output manifest at {manifest_file}")
            continue
        with open(manifest_file, "r") as f:
            filepaths.update(
                [
                    s3uri_to_relative_path(json.loads(line)["source-ref"], key_base=key_prefix)
                    for line in f
                ]
            )
    return filepaths


def stratified_sample_first_page_examples(
    input_manifest_path: str,
    n_examples: int,
    pct_first_page: float = 0.4,
    exclude_source_ref_uris: Set[str] = set(),
    random_seed: int = 1337,
) -> List[dict]:
    """Sample manifest examples, stratifying to a particular % of records with page-num 1

    Parameters
    ----------
    input_manifest_path :
        Input manifest file which should have at least keys "page-num" (1-based number) and
        "source-ref" (s3://... URI) on each record.
    n_examples :
        Number of examples to draw for the output.
    pct_first_page :
        Percentage (ratio in range 0-1) of output examples which should have page-num = 1.
    exclude_source_ref_uris :
        Set of "source-ref" values to be excluded (i.e. already-annotated images).
    random_seed :
        Random number generator initialization for reproducible draws.
    """
    with open(input_manifest_path, "r") as fmanifest:
        examples_all = [json.loads(line) for line in fmanifest]

    # Separate and shuffle the first vs non-first pages:
    examples_all_arefirsts = [item["page-num"] == 1 for item in examples_all]

    examples_firsts = [e for ix, e in enumerate(examples_all) if examples_all_arefirsts[ix]]
    examples_nonfirsts = [e for ix, e in enumerate(examples_all) if not examples_all_arefirsts[ix]]
    Random(random_seed).shuffle(examples_firsts)
    Random(random_seed).shuffle(examples_nonfirsts)

    # Exclude already-annotated images:
    filtered_firsts = [e for e in examples_firsts if e["source-ref"] not in exclude_source_ref_uris]
    filtered_nonfirsts = [
        e for e in examples_nonfirsts if e["source-ref"] not in exclude_source_ref_uris
    ]
    logger.info(
        "Excluded %s first and %s non-first pages"
        % (
            len(examples_firsts) - len(filtered_firsts),
            len(examples_nonfirsts) - len(filtered_nonfirsts),
        )
    )

    # Draw from the filtered shuffled lists:
    n_first_pages = round(pct_first_page * n_examples)
    n_nonfirst_pages = n_examples - n_first_pages
    if n_first_pages > len(filtered_firsts):
        raise ValueError(
            "Unable to find enough first-page records to build manifest: Wanted "
            "%s, but only %s available from list after exclusions (%s before)"
            % (n_first_pages, len(filtered_firsts), len(examples_firsts))
        )
    if n_nonfirst_pages > len(filtered_nonfirsts):
        raise ValueError(
            "Unable to find enough non-first-page records to build manifest: Wanted "
            "%s, but only %s available from list after exclusions (%s before)"
            % (n_nonfirst_pages, len(filtered_nonfirsts), len(examples_nonfirsts))
        )
    print(f"Taking {n_first_pages} first pages and {n_nonfirst_pages} non-first pages.")
    selected = filtered_firsts[:n_first_pages] + filtered_nonfirsts[:n_nonfirst_pages]
    Random(random_seed).shuffle(selected)  # Shuffle again to avoid putting all 1stP at front
    return selected


def consolidate_data_manifests(
    source_manifests: List[dict],
    output_manifest: TextIO,
    standard_label_field: str,
    bucket_mappings: Dict[str, str],
    prefix_mappings: Dict[str, str],
) -> None:
    """Consolidate multiple SM Ground Truth manifest files, normalizing label names and S3 URIs

    This function applies the following normalizations, which are often needed when combining
    multiple SageMaker Ground Truth manifests together into one:
    - If the jobs were created through the AWS Console with default settings, their output (labels)
        field will likely be set to the job name. A combined manifest will need to standardize
        labels to a single field name so all the records match.
    - If the jobs were created in a different AWS Account or environment, their S3 URIs may
        reference buckets and key prefixes that aren't accessible for us but are mirrored in our
        own environment. A combined manifest will need to map these S3 URIs to the current env.

    Parameters
    ----------
    source_manifests :
        List of {job_name, manifest_path} entries pointing to SageMaker Ground Truth output manifest
        files and linking them to the job names.
    output_manifest :
        Open file handle (i.e. via `open()`) for the consolidated manifest file to be written.
    standard_label_field :
        Name of the standardized label field to use in the output.
    bucket_mappings :
        Mappings from {original: replacement} S3 bucket names to replace in source manifests.
    prefix_mappings :
        Mappings from {original: replacement} S3 key prefixes to replace in source manifests.
    """
    for source in tqdm(source_manifests, desc="Consolidating manifests..."):
        job_name = source["job_name"]
        manifest_path = source["manifest_path"]
        with open(manifest_path, "r") as fin:
            for line in filter(lambda l: l, fin):
                obj: dict = json.loads(line)

                # Import refs by applying BUCKET_MAPPINGS and PREFIX_MAPPINGS:
                for k in filter(lambda k: k.endswith("-ref"), obj.keys()):
                    if not obj[k].lower().startswith("s3://"):
                        raise RuntimeError(
                            "Attr %s ends with -ref but does not start with 's3://'\n%s" % (k, obj)
                        )
                    obj_bucket, obj_key = s3uri_to_bucket_and_key(obj[k])
                    obj_bucket = bucket_mappings.get(obj_bucket, obj_bucket)
                    for old_prefix in prefix_mappings:
                        if obj_key.startswith(old_prefix):
                            obj_key = prefix_mappings[old_prefix] + obj_key[len(old_prefix) :]
                    obj[k] = f"s3://{obj_bucket}/{obj_key}"

                # Find the job output field:
                if job_name in obj:
                    source_label_attr = job_name
                elif standard_label_field in obj:
                    source_label_attr = standard_label_field
                else:
                    raise RuntimeError(
                        "Couldn't find label field for entry in {}:\n{}".format(
                            job_name,
                            obj,
                        )
                    )
                # Rename to standard:
                obj[standard_label_field] = obj.pop(source_label_attr)
                source_meta_attr = f"{source_label_attr}-metadata"
                if source_meta_attr in obj:
                    obj[f"{standard_label_field}-metadata"] = obj.pop(source_meta_attr)
                # Write to output manifest:
                output_manifest.write(json.dumps(obj) + "\n")
