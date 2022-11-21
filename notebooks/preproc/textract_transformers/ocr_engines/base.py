# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Common definitions for integrating custom OCR engines to the stack.

This module defines base classes which custom OCR integrations should conform to, and tools for
standardising OCR output in an Amazon Textract-like format.
"""
# Python Built-Ins:
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional
from uuid import uuid4

# Local Dependencies:
from ..preproc import Document


class BaseOCREngine(ABC):
    """Base class for a custom OCR engine"""

    def __init__(self, default_languages: Iterable[str]):
        """Create a BaseOCREngine

        Parameters
        ----------
        default_languages :
            List/iterable of language codes that should be detected by default in documents, in case
            `process()` method doesn't specify at run time.
        """
        self.default_languages = default_languages

    @abstractmethod
    def process(self, raw_doc: Document, languages: Optional[Iterable[str]] = None) -> dict:
        """OCR raw_doc and return Amazon Textract-like JSON

        This is the main method a custom OCR integration must implement, to process a document and
        return a compatible result. See `generate_response_json()` for output conversion tooling.
        """
        raise NotImplementedError("OCREngine must implement process() method")


def generate_guid() -> str:
    """Generate a GUID/UUID in a similar format to Amazon Textract response JSON"""
    return str(uuid4())


class OCRGeometry:
    """Amazon Textract-like object geometry data structure

    You may want to use the `from_*()` methods to create a full geometry from raw boxes/points.
    """

    def __init__(
        self, top: float, left: float, height: float, width: float, polygon: List[Dict[str, float]]
    ):
        """Create an OCRGeometry

        For this direct constructor method, you must already have a self-consistent set of both
        bounding box and polygon information. For other cases, see factory methods instead. Amazon
        Textract-like geometries use coordinates normalized to page size, so should typically be in
        the range 0-1.
        """
        if height >= 2.0 or width >= 2.0:
            raise ValueError(
                "For consistency with Amazon Textract, OCR object coordinates should be relative "
                "to page canvas and therefore approximately in range 0-1. Got height=%s, width=%s"
                % (height, width)
            )
        self.top = top
        self.left = left
        self.height = height
        self.width = width
        self.polygon = polygon

    @classmethod
    def from_bbox(cls, top: float, left: float, height: float, width: float) -> OCRGeometry:
        """Create an OCRGeometry from a (page-normalized) bounding box

        The geometry's polygon will be initialized to exactly match the bounding box. To produce an
        Amazon Textract-like result, your T/L/H/W coordinates should be relative to the page itself:
        I.e. all usually in the range 0-1.
        """
        right = left + width
        bottom = top + height
        polygon = [
            {"X": left, "Y": top},
            {"X": right, "Y": top},
            {"X": right, "Y": bottom},
            {"X": left, "Y": bottom},
        ]
        return cls(top=top, left=left, height=height, width=width, polygon=polygon)

    @classmethod
    def from_polygon_list(cls, points: List[List[float]]) -> OCRGeometry:
        """Create an OCRGeometry from a list of [x,y] coordinate tuples defining a polygon

        The geometry's bounding box will be automatically inferred from the polygon coordinates. To
        produce an Amazon Textract-like result, your T/L/H/W coordinates should be relative to the
        page itself: I.e. all usually in the range 0-1.
        """
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        top = min(x_coords)
        left = min(y_coords)
        return cls(
            top=top,
            left=left,
            height=max(y_coords) - top,
            width=max(x_coords) - left,
            polygon=[{"X": p[0], "Y": p[1]} for p in points],
        )

    @classmethod
    def union_bboxes(cls, *geometries: OCRGeometry) -> OCRGeometry:
        """Create an OCRGeometry for the box bounding several 'child' geometries

        This will produce a rectangular box, regardless of the polygon shape of input geometry
        objects.
        """
        top = min(g.top for g in geometries)
        left = min(g.left for g in geometries)
        bottom = max(g.top + g.height for g in geometries)
        right = max(g.left + g.width for g in geometries)
        return cls.from_bbox(
            top=top,
            left=left,
            height=(bottom - top),
            width=(right - left),
        )

    def to_json(self) -> dict:
        """Render this geometry as an Amazon Textract-like JSON-able dictionary"""
        return {
            "BoundingBox": {
                "Width": self.width,
                "Height": self.height,
                "Left": self.left,
                "Top": self.top,
            },
            "Polygon": [{k: v for k, v in point.items()} for point in self.polygon],
        }


class OCRWord:
    """Amazon Textract-like representation of a detected word on page"""

    def __init__(
        self, text: str, confidence: float, geometry: OCRGeometry, text_type: Optional[str] = None
    ):
        """Create an OCRWord

        TODO: Should this and other classes validate/enforce confidence>1.0?

        Parameters
        ----------
        text :
            Text of the detected word
        confidence :
            0-100 scaled confidence score for OCR of the detected word
        geometry :
            Position & shape of the detected word on the page
        text_type :
            Optional 'HANDWRITING' or 'PRINTED' specifier, if available
        """
        self.id = generate_guid()
        self.text = text
        self.confidence = confidence
        self.geometry = geometry
        self.text_type = text_type

    def to_json(self) -> dict:
        """Render this word as an Amazon Textract-like JSON-able dictionary"""
        return {
            "Id": self.id,
            "BlockType": "WORD",
            "Confidence": self.confidence,
            "Geometry": self.geometry.to_json(),
            # "Page": This will be added in post-processing by generate_response_json()
            "Text": self.text,
            **({} if self.text_type is None else {"TextType": self.text_type}),
        }


class OCRLine:
    """Amazon Textract-like representation of a line of text"""

    def __init__(
        self,
        confidence: float,
        words: Iterable[OCRWord],
        geometry: Optional[OCRGeometry] = None,
    ):
        """Create an OCRLine

        Parameters
        ----------
        confidence :
            0-100 scaled confidence with which this text line was detected
        words :
            List/iterable of word objects within this line
        geometry :
            (Optional) If a geometry for the line is not explicitly provided, the bounding box
            enclosing all the `words` will be used.
        """
        self.id = generate_guid()
        self.confidence = confidence
        self.words = words
        self._geometry = geometry

    @property
    def geometry(self) -> OCRGeometry:
        return self._geometry or OCRGeometry.union_bboxes(*(w.geometry for w in self.words))

    def to_blocks(self) -> List[dict]:
        """Render this line as list of Amazon Textract-like JSON-able blocks"""
        word_blocks = [w.to_json() for w in self.words]
        line_block = {
            "Id": self.id,
            "BlockType": "LINE",
            "Confidence": self.confidence,
            "Text": " ".join(w.text.strip() for w in self.words),
            "Geometry": self.geometry.to_json(),
            # "Page" will be added in post-processing by generate_response_json()
            "Relationships": [
                {
                    "Type": "CHILD",
                    "Ids": [word.id for word in self.words],
                },
            ],
        }
        return [line_block] + word_blocks


class OCRPage:
    """Amazon Textract-like representation of a processed page/image"""

    def __init__(self, lines: Iterable[OCRLine], geometry: Optional[OCRGeometry] = None):
        """Create an OCRPage

        Parameters
        ----------
        lines :
            List/iterable of text lines within the page
        geometry :
            Optional override geometry of the page. Defaults to a 0,0,1,1 box consistent with Amazon
            Textract.
        """
        self.id = generate_guid()
        self.geometry = geometry or OCRGeometry.from_bbox(0, 0, 1, 1)
        self.lines = lines

    def add_lines(self, lines: List[OCRLine]) -> None:
        """Add text lines to an already-created OCRPage"""
        self.lines += lines

    def to_blocks(self) -> List[dict]:
        """Render this page as list of Amazon Textract-like JSON-able blocks"""
        child_blocks = [b for line in self.lines for b in line.to_blocks()]
        page_block = {
            "Id": self.id,
            "BlockType": "PAGE",
            "Geometry": self.geometry.to_json(),
            # "Page" will be added in post-processing by generate_response_json()
            "Relationships": [
                {
                    "Type": "CHILD",
                    "Ids": [line.id for line in self.lines],
                },
            ],
        }
        return [page_block] + child_blocks


def generate_response_json(pages: List[OCRPage], engine_name: str) -> dict:
    """Create an Amazon-Textract-like, JSON-able response dict for an OCR result

    Parameters
    ----------
    pages :
        List of OCRPage objects describing detected lines and words of text with their positions
    engine_name :
        Custom OCR engine identifier, which will be reported as an alternative model version in the
        result.
    """
    page_blocks_by_page = [page.to_blocks() for page in pages]
    for page_ix, blocks in enumerate(page_blocks_by_page):
        for block in blocks:
            block["Page"] = page_ix + 1
    return {
        "DetectDocumentTextModelVersion": f"custom-{engine_name}",
        "DocumentMetadata": {"Pages": len(pages)},
        "JobStatus": "SUCCEEDED",
        "Blocks": [block for page in page_blocks_by_page for block in page],
    }
