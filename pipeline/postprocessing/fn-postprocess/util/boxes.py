# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Utilities for bounding box processing
"""
# Python Built-Ins:
from __future__ import annotations  # Self-referencing class def typing in Py3.7+
from numbers import Real
from typing import Any, Dict, Iterable, Optional, Type


EPSILON = 1e-15


class UniversalBox:
    """Box class with forgiving/flexible constructor(s) and useful props/serialization options

    No more getting bogged down translating between different formats for representing boxes!
    """

    def __init__(
        self,
        top: Optional[Real] = None,
        left: Optional[Real] = None,
        height: Optional[Real] = None,
        width: Optional[Real] = None,
        bottom: Optional[Real] = None,
        right: Optional[Real] = None,
        box: Optional[Any] = None,
        inverted_y: bool = True,
    ):
        """Create a UniversalBox from some kind of bounding box definition

        You can provide whatever sufficient combination of {top,left,height,width,bottom,right}
        keyword args you like; *OR* a `box` object with equivalent attributes - which may be
        PascalCase e.g. box.Height or box.height. `box` can also be a dict.

        `inverted_y` controls whether coordinates are image-style (default, bottom = top + height)
        or math-style (top = height + bottom).
        """

        def get_box_attr(o, attr_lower: str):
            if not o:
                return o
            if hasattr(o, attr_lower):
                return getattr(o, attr_lower)
            attr_pascal = attr_lower[0].upper() + attr_lower[1:]
            if hasattr(o, "get"):
                val = o.get(attr_lower)
                if val is None:
                    val = o.get(attr_pascal)
            else:
                val = None
            return val

        self.inverted_y = inverted_y
        self._top = get_box_attr(box, "top") if top is None else top
        self._height = get_box_attr(box, "height") if height is None else height
        self._bottom = get_box_attr(box, "bottom") if bottom is None else bottom

        self._left = get_box_attr(box, "left") if left is None else left
        self._width = get_box_attr(box, "width") if width is None else width
        self._right = get_box_attr(box, "right") if right is None else right

        if sum(map(lambda v: v is None, (self._top, self._bottom, self._height))) > 1:
            raise ValueError(
                "At least 2 of [top, height, bottom] must be specified. Got [{}, {}, {}]",
                self._top,
                self._height,
                self._bottom,
            )
        if self._top is None:
            self._top = (
                (self._bottom - self._height) if inverted_y else (self._bottom + self._height)
            )
        if self._bottom is None:
            self._bottom = (self._top + self._height) if inverted_y else (self._top - self._height)
        expected_height = (self._bottom - self._top) if inverted_y else (self._top - self._bottom)
        if self._height is None:
            self._height = expected_height
        elif abs(self._height - expected_height) > EPSILON:
            raise ValueError(
                "Specified height {} does not match specified top {} and bottom {}".format(
                    self._height,
                    self._top,
                    self._bottom,
                )
            )

        if sum(map(lambda v: v is None, (self._left, self._width, self._right))) > 1:
            raise ValueError(
                "At least 2 of [left, width, right] must be specified. Got [{}, {}, {}]",
                self._left,
                self._width,
                self._right,
            )
        if self._left is None:
            self._left = self._right - self._width
        if self._right is None:
            self._right = self._left + self._width
        expected_width = self._right - self._left
        if self._width is None:
            self._width = expected_width
        elif abs(self._width - expected_width) > EPSILON:
            raise ValueError(
                "Specified width {} does not match specified right {} - left {} = {}".format(
                    self._width,
                    self._right,
                    self._left,
                    expected_width,
                )
            )

    @property
    def top(self) -> Real:
        return self._top

    @top.setter
    def top(self, value: Real) -> None:
        self._height = self._bottom - value if self.inverted_y else self._bottom + value
        self._top = value

    @property
    def left(self) -> Real:
        return self._left

    @left.setter
    def left(self, value: Real) -> None:
        self._width = self._right - value
        self._left = value

    @property
    def height(self) -> Real:
        return self._height

    @property
    def width(self) -> Real:
        return self._width

    @property
    def bottom(self) -> Real:
        return self._bottom

    @bottom.setter
    def bottom(self, value: Real) -> None:
        self._height = self._top + value if self.inverted_y else self._top - value
        self._bottom = value

    @property
    def right(self) -> Real:
        return self._right

    @right.setter
    def right(self, value: Real) -> None:
        self._width = self._left + value
        self._right = value

    def to_dict(self, style: str = "TLHW") -> Dict[str, Real]:
        """Express the box as a (JSON serializable) dict

        Arguments
        ---------
        style : str
            Some combination of characters T,L,H,W,B,R (upper- or lower-case) indicating what
            properties should be included in the dict. E.g. 'TLbr' will generate a result with
            { 'Top', 'Left', 'bottom', 'right' }
        """
        if not style:
            return ValueError(f"Bounding box to_dict got empty style spec '{style}'")

        result = {}
        for prop in style:
            if prop == "T":
                result["Top"] = self._top
            elif prop == "t":
                result["top"] = self._top
            elif prop == "L":
                result["Left"] = self._left
            elif prop == "l":
                result["left"] = self._left
            elif prop == "H":
                result["Height"] = self._height
            elif prop == "h":
                result["height"] = self._height
            elif prop == "W":
                result["Width"] = self._width
            elif prop == "w":
                result["width"] = self._width
            elif prop == "B":
                result["Bottom"] = self._bottom
            elif prop == "b":
                result["bottom"] = self._bottom
            elif prop == "R":
                result["Right"] = self._right
            elif prop == "r":
                result["right"] = self._right
            else:
                raise ValueError(
                    f"Bounding box to_dict style '{style}' contained unrecognised spec '{prop}'"
                )
        return result

    @classmethod
    def aggregate(
        cls: Type[UniversalBox],
        boxes: Iterable[UniversalBox],
        inverted_y: Optional[bool] = None,
    ) -> UniversalBox:
        """Calculate the minimal bounding box containing input `boxes`

        Arguments
        ---------
        boxes : Iterable[UniversalBox]
            The UniversalBox instances to combine
        inverted_y : Optional[bool]
            If not provided, will be inferred from the `boxes`
        """
        if not (boxes and len(boxes)):
            raise ValueError(f"Cannot aggregate with no 'boxes'! Got {boxes}")

        if inverted_y is None:
            n_inverted_ys = sum(b.inverted_y for b in boxes)
            inverted_y = n_inverted_ys > (len(boxes) / 2)

        box_tops = [b.top if b.inverted_y == inverted_y else b.bottom for b in boxes]
        box_bottoms = [b.bottom if b.inverted_y == inverted_y else b.top for b in boxes]
        return cls(
            top=min(box_tops) if inverted_y else max(box_tops),
            bottom=max(box_bottoms) if inverted_y else min(box_bottoms),
            left=min(b.left for b in boxes),
            right=max(b.right for b in boxes),
            inverted_y=inverted_y,
        )
