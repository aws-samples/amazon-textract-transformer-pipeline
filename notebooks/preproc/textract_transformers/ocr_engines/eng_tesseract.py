# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Example integration for (Py)Tesseract as a custom OCR engine
"""
# Python Built-Ins:
from logging import getLogger
import os
from statistics import mean
from tempfile import TemporaryDirectory
from typing import Iterable, List, Optional

# External Dependencies:
import pandas as pd
import pytesseract

# Local Dependencies:
from .base import BaseOCREngine, generate_response_json, OCRGeometry, OCRLine, OCRPage, OCRWord
from ..image_utils import Document


logger = getLogger("eng_tesseract")


if os.environ.get("TESSDATA_PREFIX") is None:
    os.environ["TESSDATA_PREFIX"] = "/opt/conda/share/tessdata"


class TesseractEngine(BaseOCREngine):
    """Tesseract-based engine for custom SageMaker OCR endpoint option"""

    engine_name = "tesseract"

    def process(self, raw_doc: Document, languages: Optional[Iterable[str]] = None) -> dict:
        ocr_pages = []

        with TemporaryDirectory() as tmpdir:
            raw_doc.set_workspace(tmpdir)
            for ixpage, page in enumerate(raw_doc.get_pages()):
                logger.debug(f"Serializing page {ixpage + 1}")
                page_ocr = pytesseract.image_to_data(
                    page.file_path,
                    output_type=pytesseract.Output.DATAFRAME,
                    lang="+".join(self.default_languages if languages is None else languages),
                    pandas_config={
                        # Need this explicit override or else pages containing only a single number
                        # can sometimes have text column interpreted as numeric type:
                        "dtype": {"text": str},
                    },
                )
                ocr_pages += self.dataframe_to_ocrpages(page_ocr)
        return generate_response_json(ocr_pages, self.engine_name)

    @classmethod
    def dataframe_to_ocrpages(cls, ocr_df: pd.DataFrame) -> List[OCRPage]:
        """Convert a Tesseract DataFrame to a list of OCRPage ready for Textract-like serialization

        Tesseract TSVs / PyTesseract DataFrames group detections by multiple levels: Page, block,
        paragraph, line, word. Columns are: level, page_num, block_num, par_num, line_num, word_num,
        left, top, width, height, conf, text.

        Each level is introduced by a record, so for example there will be an initial record with
        (level=1, page_num=1, block_num=0, par_num=0, line_num=0, word_num=0)... And then several
        others before finally getting down to the first WORD record (level=5, page_num=1,
        block_num=1, par_num=1, line_num=1, word_num=1). Records are assumed to be sorted in order,
        as indeed they are direct from Tesseract.
        """
        # First construct an indexable list of page geometries, as we'll need these to normalize
        # other entity coordinates from absolute pixel values to 0-1 range:
        # (Note: In fact this function will often be called with only one page_num at a time)
        page_dims = (
            ocr_df[ocr_df["level"] == 1]
            .groupby("page_num")
            .agg(
                {
                    "left": "min",
                    "top": "min",
                    "width": "max",
                    "height": "max",
                    "page_num": "count",
                }
            )
        )
        # There should be exactly one level=1 record per page in the dataframe. After checking
        # this, we can dispose the "page_num" count column.
        if (page_dims["page_num"] > 1).sum() > 0:
            raise ValueError(
                "Tesseract DataFrame had duplicate entries for these page_nums at level 1: %s"
                % page_dims.index[page_dims["page_num"] > 0].values[:20]
            )
        page_dims.drop(columns="page_num", inplace=True)

        # We need to collapse the {block, paragraph} levels of Tesseract hierarchy to preserve only
        # PAGE, LINE and WORD for consistency with Textract. Here we'll assume the DataFrame is in
        # its original Tesseract sort order, allowing iteration through the records to correctly
        # roll the entities up. Although iterating through large DataFrames isn't generally a
        # performant practice, this could always be balanced with specific parallelism if wanted:
        # E.g. processing multiple pages at once.
        pages = {
            num: OCRPage([])  # Initialise all pages first with no text
            for num in sorted(ocr_df[ocr_df["level"] == 1]["page_num"].unique())
        }
        cur_page_num = None
        page_lines = []
        cur_line_id = None
        line_words = []

        # Tesseract LINE records (level 4) don't have a confidence (equals -1), so we'll use the
        # average over the included WORDs as a heuristic. They *do* have T/L/H/W geometry info, but
        # we'll ignore that for the sake of code simplicity and let OCRLine infer it from the union
        # of all WORD bounding boxes.
        add_line = lambda words: (
            page_lines.append(OCRLine(mean(w.confidence for w in words), words))
        )

        # Loop through all WORD records, ignoring whitespace-only ones that Tesseract likes to yield
        words_df = ocr_df[ocr_df["level"] == 5].copy()
        words_df["text"] = words_df["text"].str.strip()
        words_df = words_df[words_df["text"].str.len() > 0]
        for _, rec in words_df.iterrows():
            line_id = (rec.block_num, rec.par_num, rec.line_num)
            page_num = rec.page_num
            if cur_line_id != line_id:
                # Start of new LINE - add previous one to result:
                if cur_line_id is not None:
                    add_line(line_words)
                cur_line_id = line_id
                line_words = []
            if cur_page_num != page_num:
                # Start of new PAGE - add previous one to result:
                if cur_page_num is not None:
                    pages[cur_page_num].add_lines(page_lines)
                cur_page_num = page_num
                page_lines = []
            # Parse this record into a WORD:
            page_dim_rec = page_dims.loc[page_num]
            line_words.append(
                OCRWord(
                    rec.text,
                    rec.conf,
                    OCRGeometry.from_bbox(
                        # Word geometries, too, need normalizing by page dimensions.
                        (rec.top - page_dim_rec.top) / page_dim_rec.height,
                        (rec.left - page_dim_rec.left) / page_dim_rec.width,
                        rec.height / page_dim_rec.height,
                        rec.width / page_dim_rec.width,
                    ),
                )
            )
        # End of last line and last page: Add any remaining content.
        if len(line_words):
            add_line(line_words)
        if len(page_lines):
            pages[cur_page_num].add_lines(page_lines)
        return [page for page in pages.values()]
