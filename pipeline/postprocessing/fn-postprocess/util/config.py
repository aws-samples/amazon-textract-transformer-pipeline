# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Document field/entity configuration definition utilities
"""
# Python Built-Ins:
from enum import Enum
from typing import Callable, Optional

# Local Dependencies:
from .deser import PascalJsonableDataClass


class FieldSelectionMethod:
    def __init__(self, name: str, sort: Callable, desc: bool = False):
        self.name = name
        self.sort = sort
        self.desc = desc

    def to_dict(self):
        return self.name


class FieldSelectionMethods(Enum):
    CONFIDENCE = FieldSelectionMethod("confidence", lambda v: v["Confidence"], desc=True)
    FIRST = FieldSelectionMethod("first", lambda v: v["IxFirstDetection"])
    LAST = FieldSelectionMethod("last", lambda v: v["IxLastDetection"], desc=True)
    LONGEST = FieldSelectionMethod("longest", lambda v: len(v["Text"]), desc=True)
    SHORTEST = FieldSelectionMethod("shortest", lambda v: len(v["Text"]))


class FieldConfiguration(PascalJsonableDataClass):
    """A JSON-serializable configuration for a field/entity type"""

    def __init__(
        self,
        class_id: int,
        name: str,
        ignore: Optional[bool] = None,
        optional: Optional[bool] = None,
        select: Optional[str] = None,
        annotation_guidance: Optional[str] = None,
        normalizer_endpoint: Optional[str] = None,
        normalizer_prompt: Optional[str] = None,
    ):
        """Create a FieldConfiguration

        Parameters
        ----------
        class_id : int
            The ID number (ordinal) of the class per the machine learning model
        name : str
            The human-readable name of the class / entity type
        ignore : Optional[bool]
            Set True to exclude this field from post-processing in the OCR pipeline (the ML model
            will still be trained on it). Useful if for e.g. testing a new field type with unknown
            detection quality.
        optional : Optional[bool]
            Set True to explicitly indicate the field is optional (default None)
        select : Optional[str]
            A (case insensitive) name from the FieldSelectionMethods enum (e.g. 'confidence') to
            indicate how the "winning" detected value of a field should be selected. If omitted,
            the field is treated as multi-value and all detected values passed through.
        annotation_guidance : Optional[str]
            HTML-tagged guidance detailing the specific scope for this entity: I.e. what should
            and should not be included for consistent labelling.
        normalizer_endpoint : Optional[str]
            An optional deployed SageMaker seq2seq endpoint for field value normalization, if one
            should be used (You'll have to train and deploy this endpoint separately).
        normalizer_prompt : Optional[str]
            The prompting prefix for the seq2seq field value normalization requests on this field,
            if enabled. For example, "Convert dates to YYYY-MM-DD: "
        """
        self.class_id = class_id
        self.name = name
        self.ignore = ignore
        self.optional = optional
        self.annotation_guidance = annotation_guidance
        self.normalizer_endpoint = normalizer_endpoint
        self.normalizer_prompt = normalizer_prompt
        try:
            self.select = FieldSelectionMethods[select.upper()].value if select else None
        except KeyError as e:
            raise ValueError(
                "Selection method '{}' configured for field '{}' not in the known list {}".format(
                    select,
                    name,
                    [fsm.name for fsm in FieldSelectionMethods],
                )
            ) from e
        if bool(self.normalizer_endpoint) ^ bool(self.normalizer_prompt):
            raise ValueError(
                "Cannot provide only one of `normalizer_endpoint` and `normalizer_prompt` without "
                "setting both. Got: '%s' and '%s'"
                % (self.normalizer_endpoint, self.normalizer_prompt)
            )
