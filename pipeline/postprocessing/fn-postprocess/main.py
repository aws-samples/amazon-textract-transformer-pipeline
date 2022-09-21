# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Lambda to extract business fields from SageMaker-enriched Textract result

Env var DEFAULT_ENTITY_CONFIG (or the contents of SSM parameter whose name is given by
DEFAULT_ENTITY_CONFIG_PARAM) should be a JSON list of objects roughly as:

```python
[
    {
        "ClassId": 0,  # (int, required) ID of the class per SageMaker model
        "Name": "...",  # (str, required) Human-readable name of the entity/field
        "Ignore": True, # (bool, optional) Set true to ignore detections of this field
        "Optional": True,  # (bool, optional) Set true to indicate param is optional
        "Select": "..." # (str, optional) name util.config.FieldSelectors
    }
]
```

If "Select" is not specified, the field is determined to be multi-value.

For full entity configuration details, refer to util.config.FieldConfiguration
"""

# Python Built-Ins:
from functools import reduce
import json
import logging
import os
from typing import List

# External Dependencies:
import boto3  # General-purpose AWS SDK for Python
import trp  # Amazon Textract Response Parser

# Local Dependencies
from util.boxes import UniversalBox
from util.config import FieldConfiguration


logger = logging.getLogger()
logger.setLevel(logging.INFO)
s3 = boto3.resource("s3")
ssm = boto3.client("ssm")

DEFAULT_ENTITY_CONFIG = os.environ.get("DEFAULT_ENTITY_CONFIG")
if DEFAULT_ENTITY_CONFIG is not None:
    DEFAULT_ENTITY_CONFIG = json.loads(DEFAULT_ENTITY_CONFIG)
DEFAULT_ENTITY_CONFIG_PARAM = os.environ.get("DEFAULT_ENTITY_CONFIG_PARAM")


class MalformedRequest(ValueError):
    pass


def handler(event, context):
    try:
        srcbucket = event["Input"]["Bucket"]
        srckey = event["Input"]["Key"]
        entity_config = event.get("EntityConfig", DEFAULT_ENTITY_CONFIG)
        if entity_config is None and DEFAULT_ENTITY_CONFIG_PARAM:
            entity_config = json.loads(
                ssm.get_parameter(Name=DEFAULT_ENTITY_CONFIG_PARAM)["Parameter"]["Value"]
            )
    except KeyError as ke:
        raise MalformedRequest(f"Missing field {ke}, please check your input payload") from ke
    if entity_config is None:
        raise MalformedRequest(
            "Request did not specify EntityConfig, and neither env var DEFAULT_ENTITY_CONFIG (for "
            "inline json) nor DEFAULT_ENTITY_CONFIG_PARAM (for SSM parameter) are set"
        )
    entity_config = [FieldConfiguration.from_dict(cfg) for cfg in entity_config]

    doc = json.loads(s3.Bucket(srcbucket).Object(srckey).get()["Body"].read())
    doc = trp.Document(doc)

    entities = extract_entities(doc, entity_config)

    result_fields = {}
    for ixtype, cfg in enumerate(cfg for cfg in entity_config if not cfg.ignore):
        # Filter the list of detected entity mentions for this class only:
        field_entities = list(filter(lambda e: e.cls_id == cfg.class_id, entities))

        # Consolidate multiple detections of exactly the same value (text):
        field_values = {}
        for ixe, e in enumerate(field_entities):
            if e.text in field_values:
                field_values[e.text]["Detections"].append(e)
                field_values[e.text]["IxLastDetection"] = ixe
            else:
                field_values[e.text] = {
                    "Text": e.text,
                    "Detections": [e],
                    "IxFirstDetection": ixe,
                    "IxLastDetection": ixe,
                }
        field_values_list = [v for v in field_values.values()]
        # To approximate confidence for values detected multiple times, model each detection as an
        # uncorrelated observation of that value (naive, probably biased to over-estimate):
        for v in field_values_list:
            # e.g. {0.84, 0.86, 0.90} -> 1 - (0.16 * 0.14 * 0.1) = 0.998
            v["Confidence"] = 1 - reduce(
                lambda acc, next: acc * (1 - next.confidence),
                v["Detections"],
                1.0,
            )
        # TODO: Adjust for other (disagreeing) confidences better
        value_conf_norm = reduce(lambda acc, next: acc + next["Confidence"], field_values_list, 0.0)
        for v in field_values_list:
            v["Confidence"] = v["Confidence"] / max(1.0, value_conf_norm)

        field_result = {
            "ClassId": cfg.class_id,
            "Confidence": 0.0,
            "NumDetections": len(field_entities),
            "NumDetectedValues": len(field_values),
            "SortOrder": ixtype,
        }
        result_fields[cfg.name] = field_result
        if cfg.optional is not None:
            field_result["Optional"] = cfg.optional

        if cfg.select is not None:
            # Single-valued field: Select 'best' matched values:
            selector = cfg.select
            field_values_sorted = sorted(
                field_values_list,
                key=selector.sort,
                reverse=selector.desc,
            )
            if len(field_values_sorted):
                field_result["Value"] = field_values_sorted[0]["Text"]
                field_result["Confidence"] = field_values_sorted[0]["Confidence"]
                field_result["Detections"] = list(
                    map(
                        lambda e: e.to_dict(),
                        field_values_sorted[0]["Detections"],
                    )
                )
            else:
                field_result["Value"] = ""
                field_result["Detections"] = []
        else:
            # Multi-valued field: Pass through all matched values
            field_result["Values"] = list(
                map(
                    lambda v: {
                        "Confidence": v["Confidence"],
                        "Value": v["Text"],
                        "Detections": list(
                            map(
                                lambda e: e.to_dict(),
                                v["Detections"],
                            )
                        ),
                    },
                    sorted(field_values_list, key=lambda v: v["Confidence"], reverse=True),
                )
            )
            if len(field_result["Values"]):
                # For multi value, take field confidence = average value confidence
                field_result["Confidence"] = reduce(
                    lambda acc, next: acc + next["Confidence"],
                    field_result["Values"],
                    0.0,
                ) / len(field_result["Values"])

    return {
        "Confidence": min(
            r["Confidence"]
            for r in result_fields.values()
            if not (r["Confidence"] == 0 and r.get("Optional"))
        ),
        "Fields": result_fields,
    }


class EntityDetection:
    def __init__(self, trp_words, cls_id: int, cls_name: str, page_num: int):
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

    def to_dict(self):
        return {
            "ClassId": self.cls_id,
            "ClassName": self.cls_name,
            "Confidence": self.confidence,
            "Blocks": self.blocks,
            "BoundingBox": self.bbox.to_dict(),
            "PageNum": self.page_num,
            "Text": self.text,
        }

    def __repr__(self):
        return json.dumps(self.to_dict())


def extract_entities(
    doc: trp.Document,
    entity_config: List[FieldConfiguration],
) -> List[EntityDetection]:
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
