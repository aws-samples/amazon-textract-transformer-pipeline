# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Data pre-processing and manifest consolidation utilities
"""
# Python Built-Ins:
from __future__ import annotations
import json
from logging import getLogger
import re
from typing import Iterable, List, Optional, Tuple, Union

# External Dependencies:
import boto3
from sagemaker.estimator import Framework
from tqdm.notebook import tqdm  # Progress bars
import trp  # Amazon Textract Response Parser

# Local Dependencies:
from .s3 import s3_object_exists, s3uri_to_bucket_and_key


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
