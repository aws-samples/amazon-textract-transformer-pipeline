# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Utils to extract entity mentions from SageMaker Textract WORD-tagging model results

As a simple heuristic, consecutive WORD blocks of the same tagged entity class are tagged as
belonging to the same mention. This means that in cases where the normal human reading order
diverges from the Amazon Textract block output order, mentions may get split up.
"""
# Python Built-Ins:
import json
from typing import List, Optional, Sequence

# External Dependencies:
import trp  # Amazon Textract Response Parser

# Local Dependencies:
from .boxes import UniversalBox
from .config import FieldConfiguration


class EntityDetection:
    """Object describing an entity mention in a document

    If property `raw_text` (or 'RawText' in the JSON-ified equivalent) is set, this mention has
    been normalized. Otherwise, `text` is as per the original document.
    """

    raw_text: Optional[str]

    def __init__(self, trp_words: Sequence[trp.Word], cls_id: int, cls_name: str, page_num: int):
        self.cls_id = cls_id
        self.cls_name = cls_name
        self.page_num = page_num

        if len(trp_words) and not hasattr(trp_words[0], "id"):
            trp_words_by_line = trp_words
            trp_words_flat = [w for ws in trp_words for w in ws]

        else:
            trp_words_by_line = [trp_words]
            trp_words_flat = trp_words
        self.bbox = UniversalBox.aggregate(
            boxes=[UniversalBox(box=w.geometry.boundingBox) for w in trp_words_flat],
        )
        self.blocks = list(map(lambda w: w.id, trp_words_flat))
        self.confidence = min(
            map(
                lambda w: min(
                    w._block.get("PredictedClassConfidence", 1.0),
                    w.confidence,
                ),
                trp_words_flat,
            )
        )
        self.text = "\n".join(
            map(
                lambda words: " ".join([w.text for w in words]),
                trp_words_by_line,
            )
        )
        self.raw_text = None

    def normalize(self, normalized_text: str) -> None:
        """Update the detection with a new normalized text value

        Only the original raw_text value will be preserved, so if you normalize() multiple times no
        record of the intermediate normalized_text values will be kept.
        """
        if self.raw_text is None:
            self.raw_text = self.text
        # Otherwise keep original 'raw' text (normalize called multiple times)
        self.text = normalized_text

    def to_dict(self) -> dict:
        """Represent this mention as a PascalCase JSON-able object"""
        result = {
            "ClassId": self.cls_id,
            "ClassName": self.cls_name,
            "Confidence": self.confidence,
            "Blocks": self.blocks,
            "BoundingBox": self.bbox.to_dict(),
            "PageNum": self.page_num,
            "Text": self.text,
        }
        if self.raw_text is not None:
            result["RawText"] = self.raw_text
        return result

    def __repr__(self) -> str:
        return json.dumps(self.to_dict())


def extract_entities(
    doc: trp.Document,
    entity_config: List[FieldConfiguration],
) -> List[EntityDetection]:
    """Collect EntityDetections from an NER-enriched Textract JSON doc into a flat list"""
    entity_classes = {c.class_id: c.name for c in entity_config if not c.ignore}
    detections = []

    current_cls = None
    current_entity = []
    for ixpage, page in enumerate(doc.pages):
        for line in page.lines:  # TODO: Lines InReadingOrder?
            current_entity.append([])
            for word in line.words:
                pred_cls = word._block.get("PredictedClass")
                if pred_cls not in entity_classes:
                    pred_cls = None  # Treat all non-config'd entities as "other"

                if pred_cls != current_cls:
                    if current_cls is not None:
                        detections.append(
                            EntityDetection(
                                trp_words=list(
                                    filter(
                                        lambda ws: len(ws),
                                        current_entity,
                                    )
                                ),
                                cls_id=current_cls,
                                cls_name=entity_classes[current_cls],
                                page_num=ixpage + 1,
                            )
                        )
                    current_cls = pred_cls
                    current_entity = [[]] if pred_cls is None else [[word]]
                elif pred_cls is not None:
                    current_entity[-1].append(word)

    return detections
