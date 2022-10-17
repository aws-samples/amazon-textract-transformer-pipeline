# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Custom/open-source OCR engine integrations

Use .get() defined in this __init__.py file, to dynamically load your custom engine(s).

To be discoverable by get(), your module (script or folder):

- Should be placed in this folder, with a name beginning 'eng_'.
- Should expose a class inheriting from `base.BaseOCREngine`, preferably with a name
    ending with 'Engine', and should not expose multiple such classes.
"""
# Python Built-Ins:
from importlib import import_module
import inspect
from logging import getLogger
import os
from types import ModuleType
from typing import Dict, Iterable, Type

# Local Dependencies:
from .base import BaseOCREngine


logger = getLogger("ocr_engines")


# Auto-discover all eng_*** modules in this folder as [EngineName->ModuleName]:
ENGINES: Dict[str, str] = {}
for item in os.listdir(os.path.dirname(__file__)):
    if not item.startswith("eng_"):
        continue
    if item.endswith(".py"):
        # (Assuming everything starting with eng_ is a folder or a .py file), strip ext if present:
        item = item[: -len(".py")]
    name = item[len("eng_") :]  # ID/Name of the engine strips leading 'eng_'
    ENGINES[name] = "." + item  # Relative importable module name


def _find_ocr_engine_class(module: ModuleType) -> Type[BaseOCREngine]:
    """Find the OCREngine class from an imported module"""
    class_names = [name for name in dir(module) if inspect.isclass(module.__dict__[name])]
    names_ending_engine = [n for n in dir(module) if n.endswith("Engine")]
    engine_child_classes = [
        name
        for name in class_names
        if issubclass(module.__dict__[name], BaseOCREngine)
        and module.__dict__[name] is not BaseOCREngine
    ]
    preferred_names = [n for n in engine_child_classes if n in names_ending_engine]

    if len(preferred_names) == 1:
        name = preferred_names[0]
    elif len(engine_child_classes) == 1:
        name = engine_child_classes[0]
    elif len(names_ending_engine) == 1:
        name = names_ending_engine[0]
    elif len(class_names) == 1:
        name = class_names[0]
    else:
        raise ImportError(
            "Failed to find unique BaseOCREngine child class from OCR engine module '%s'. Classes "
            "inheriting from BaseOCREngine: %s. Class names defined by module: %s"
            % (module.__name__, engine_child_classes, class_names)
        )
    return module.__dict__[name]


def get(engine_name: str, default_languages: Iterable[str]) -> BaseOCREngine:
    """Initialize a supported custom OCR engine by name

    Engines are dynamically imported, so that ImportErrors aren't raised for missing dependencies
    unless there's an actual attempt to create/use the engine.

    Parameters
    ----------
    engine_name :
        Name of a supported custom OCR engine to fetch.
    default_languages :
        Language codes to configure the engine to detect by default.
    """
    if engine_name in ENGINES:
        # Load the module:
        logger.info("Loading OCR engine '%s' from module '%s'", engine_name, ENGINES[engine_name])
        module = import_module(ENGINES[engine_name], package=__name__)
        # Locate the target class in the module:
        cls = _find_ocr_engine_class(module)
        logger.info("Loading engine class: %s", cls)
        # Load the engine:
        return cls(default_languages)
    else:
        raise ValueError(
            "Couldn't find engine '%s' in ocr_engines module. Not in set: %s"
            % (engine_name, ENGINES)
        )
