# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Utilities for de/serializing typed Python objects to JSON-able dicts and actual JSON strings.
"""
# Python Built-Ins:
import json
import re
from typing import Iterable, Optional


def pascal_to_snake_case(s: str) -> str:
    """Convert a string from PascalCase to snake_case"""
    if not s:
        return s
    # Interpret sequences of 2+ uppercase chars as acronyms to be wordified. e.g. HTML -> Html
    result = re.sub(r"([A-Z])([A-Z]+)", lambda m: m.group(1) + m.group(2).lower(), s)
    # Find any uppercase character(s) following a lowercase character, insert an underscore and
    # convert. E.g. aA -> a_a, MyHtmlThing -> my_html_thing
    # Replace any aA combo with a_a:
    result = re.sub(
        r"([^A-Z])([A-Z]+)",
        lambda m: "_".join((m.group(1), m.group(2).lower())),
        result,
    )
    # Force lowercase first char:
    return result[0].lower() + result[1:]


def snake_to_pascal_case(s: str) -> str:
    """Convert a string from snake_case to PascalCase"""
    if not s:
        return s
    return "".join(
        map(
            lambda segment: (segment[0].upper() + segment[1:]) if segment else segment,
            s.split("_"),
        ),
    )


class PascalJsonableDataClass:
    """Mixin to make a class with snake_case attrs interop with JSON/dicts with PascalCase attrs

    from_dict maps dict keys to constructor args { "MyProp": 1 } -> __init__(my_prop=1)

    to_dict maps data properties (as enumerated by __dict__) to dict keys

    from_json/to_json methods simply wrap the above with json.loads() / json.dumps()
    """

    @classmethod
    def from_dict(cls, d: dict):
        kwargs = {pascal_to_snake_case(k): v for k, v in d.items()}
        return cls(**kwargs)

    @classmethod
    def from_json(cls, s: str):
        return cls.from_dict(json.loads(s))

    def to_dict(self, omit: Optional[Iterable[str]] = None):
        if not omit:
            omit = []
        return {
            snake_to_pascal_case(attr): value.to_dict() if hasattr(value, "to_dict") else value
            for attr, value in filter(
                lambda kv: not (kv[1] is None or kv[0].startswith("_") or kv[0] in omit),
                self.__dict__.items(),
            )
        }

    def to_json(self, omit: Optional[Iterable[str]] = None):
        return json.dumps(self.to_dict(omit=omit))
