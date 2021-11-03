# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""SageMaker Processing script to prepare raw documents into SMGT-ready page images
"""

# Python Built-Ins:
import argparse
import json
import logging
from multiprocessing import cpu_count, Pool
import os
import shutil
import time
from types import SimpleNamespace
from typing import Iterable, List, Optional

logging.basicConfig(level="INFO", format="%(asctime)s %(name)s [%(levelname)s] %(message)s")

# External Dependencies:
import pdf2image
import PIL
from PIL import ExifTags


logger = logging.getLogger("preproc")


def split_filename(filename):
    basename, _, ext = filename.rpartition(".")
    return basename, ext


class ImageExtractionResult:
    """Result descriptor for extracting a source image/document to image(s)"""

    def __init__(self, rawpath: str, cleanpaths: List[str] = [], cats: List[str] = []):
        self.rawpath = rawpath
        self.cleanpaths = cleanpaths
        self.cats = cats


def clean_dataset_for_img_ocr(
    from_path: str,
    to_path: str,
    filepaths: Optional[Iterable[str]] = None,
    pdf_dpi: int = 300,
    pdf_image_format: str = "png",
    textract_compatible_formats: Iterable[str] = ("jpg", "jpeg", "png"),
    preferred_image_format: str = "png",
) -> List[ImageExtractionResult]:
    """Process a mixed PDF/image dataset for use with SageMaker Ground Truth image task UIs

    Extracts page images from PDFs, converts EXIF-rotated images to data-rotated.

    Parameters
    ----------
    from_path : str
        Base path of the raw/source dataset to be converted
    to_path : str
        Target path for converted files (subfolder structure will be preserved from source)
    filepaths : Optional[Iterable[str]]
        Paths (relative to from_path, no leading slash) to filter down the processing. If not
        provided, the whole from_path folder will be recursively crawled.
    pdf_dpi : int
        DPI resolution to extract images from PDFs (Default 300).
    pdf_image_format : str
        Format to extract images from PDFs (Default 'png').
    textract_compatible_formats : Iterable[str]
        The set of compatible file formats for Textract: Used to determine whether to convert
        source images in other formats which PIL may still have been able to successfully load.
    preferred_image_format : str
        Format to be used when an image has been saved/converted (Default 'png').
    """
    results = []
    if filepaths:
        # Users will supply relative filepaths, but our code below expects to include from_path:
        filepaths = [os.path.join(from_path, p) for p in filepaths]
    else:
        # List input files automatically by walking the from_path dir:
        filepaths = list(
            filter(
                lambda path: "/." not in path,  # Ignore hidden stuff e.g. .ipynb_checkpoints
                (os.path.join(path, f) for path, _, files in os.walk(from_path) for f in files),
            )
        )
    n_files_total = len(filepaths)
    os.makedirs(to_path, exist_ok=True)

    for ixfilepath, filepath in enumerate(filepaths):
        # TODO: Would be better to have global progress logging than per-worker
        logger.debug("Doc %s of %s", ixfilepath + 1, len(filepaths))
        # Because we use all available cores, things can become unhappy if we don't provide a
        # little breathing space for any background procesess:
        time.sleep(0.5)
        
        filename = os.path.basename(filepath)
        subfolder = os.path.dirname(filepath)[len(from_path) :]  # Includes leading slash!
        outfolder = to_path + subfolder
        os.makedirs(outfolder, exist_ok=True)
        basename, ext = split_filename(filename)
        ext_lower = ext.lower()
        result = ImageExtractionResult(
            rawpath=filepath,
            cleanpaths=[],
            cats=subfolder[1:].split(os.path.sep),  # Strip leading slash to avoid initial ''
        )
        if ext_lower == "pdf":
            logger.info(
                "Converting {} to {}/{}*.{}".format(
                    filepath,
                    outfolder,
                    basename + "-",
                    pdf_image_format,
                )
            )
            images = pdf2image.convert_from_path(
                filepath,
                output_folder=outfolder,
                output_file=basename + "-",
                # TODO: Use paths_only option to return paths instead of image objs
                fmt=pdf_image_format,
                dpi=pdf_dpi,
            )
            result.cleanpaths = [i.filename for i in images]
            results.append(result)
            logger.info(
                "* PDF converted {}:\n    - {}".format(
                    filepath,
                    "\n    - ".join(result.cleanpaths),
                )
            )
        else:
            try:
                image = PIL.Image.open(filepath)
            except PIL.UnidentifiedImageError:
                logger.warning(f"* Ignoring incompatible file: {filepath}")
                continue

            # Correct orientation from EXIF data:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == "Orientation":
                    break
            exif = dict((image._getexif() or {}).items())
            img_orientation = exif.get(orientation)
            logger.info("Image {} has orientation {}".format(filepath, img_orientation))
            if img_orientation == 3:
                image = image.rotate(180, expand=True)
                rotated = True
            elif img_orientation == 6:
                image = image.rotate(270, expand=True)
                rotated = True
            elif img_orientation == 8:
                image = image.rotate(90, expand=True)
                rotated = True
            else:
                rotated = False

            if ext_lower not in textract_compatible_formats:
                outpath = os.path.join(outfolder, f"{basename}.{preferred_image_format}")
                image.save(outpath)
                logger.info(f"* Converted image {filepath} to {outpath}")
            elif rotated:
                outpath = os.path.join(outfolder, filename)
                image.save(outpath)
                logger.info(f"* Rotated image {filepath} to {outpath}")
            else:
                outpath = os.path.join(outfolder, filename)

                shutil.copy2(filepath, outpath)
                logger.info(f"* Copied file {filepath} to {outpath}")
            result.cleanpaths = [outpath]
            results.append(result)

    logger.info("Done!")
    return results


def parse_args():
    """Parse SageMaker Processing Job CLI arguments to job parameters
    """
    parser = argparse.ArgumentParser(
        description="Split and standardize documents into images for OCR"
    )
    parser.add_argument("--input", type=str, default="/opt/ml/processing/input/raw",
        help="Folder where raw input images/documents are stored"
    )
    parser.add_argument("--output", type=str, default="/opt/ml/processing/output/imgs-clean",
        help="Folder where cleaned output images should be saved"
    )
    parser.add_argument("--n-workers", type=int, default=cpu_count(),
        help="Number of worker processes to use for extraction (default #CPU cores)"
    )
    args = parser.parse_args()
    return args


def process_shard(inputs: dict):
    return clean_dataset_for_img_ocr(**inputs)


if __name__ == "__main__":
    # Main processing job script:
    args = parse_args()

    logger.info(f"Reading raw files from {args.input}")
    rel_filepaths_all = sorted(filter(
        lambda f: not (f.startswith(".") or "/." in f), # (Exclude hidden dot-files)
        [
            os.path.join(currpath, name)[len(args.input) + 1:]  # +1 for trailing '/'
            for currpath, dirs, files in os.walk(args.input)
            for name in files
        ]
    ))
    logger.info(rel_filepaths_all[:10])
    
    logger.info(f"Processing {len(rel_filepaths_all)} files across {args.n_workers} processes")
    with Pool(args.n_workers) as pool:
        # TODO: Test pooling individual samples rather than list slices
        # This would likely increase inter-process-communication overheads, but reduce risk of one
        # process taking much longer to complete than others.
        pool.map(
            process_shard,
            [
                {
                    "from_path": args.input,
                    "to_path": args.output,
                    "filepaths": rel_filepaths_all[ix::args.n_workers],
                }
                for ix in range(args.n_workers)
            ],
        )
    logger.info("All done!")
