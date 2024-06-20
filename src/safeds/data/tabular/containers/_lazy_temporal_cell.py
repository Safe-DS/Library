from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash

from ._lazy_cell import _LazyCell
from ._temporal_cell import TemporalCell

if TYPE_CHECKING:
    import polars as pl

    from ._cell import Cell


class _LazyTemporalCell(TemporalCell):
    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, expression: pl.Expr) -> None:
        self._expression: pl.Expr = expression

    def __hash__(self) -> int:
        return _structural_hash(self._expression.meta.serialize())

    def __sizeof__(self) -> int:
        return self._expression.__sizeof__()

    # ------------------------------------------------------------------------------------------------------------------
    # Temporal operations
    # ------------------------------------------------------------------------------------------------------------------

    def century(self) -> Cell[int]:
        return _LazyCell(self._expression.dt.century())

    def weekday(self) -> Cell[int]:
        return _LazyCell(self._expression.dt.weekday())

    def week(self) -> Cell[int]:
        return _LazyCell(self._expression.dt.week())

    def year(self) -> Cell[int]:
        return _LazyCell(self._expression.dt.year())

    def month(self) -> Cell[int]:
        return _LazyCell(self._expression.dt.month())

    def day(self) -> Cell[int]:
        return _LazyCell(self._expression.dt.day())

    def datetime_to_string(self, format_string: str = "%Y/%m/%d %H:%M:%S") -> Cell[str]:
        if not _check_format_string(format_string):
            raise ValueError("Invalid format string")
        return _LazyCell(self._expression.dt.to_string(format=format_string))

    def date_to_string(self, format_string: str = "%F") -> Cell[str]:
        if not _check_format_string(format_string):
            # Fehler in _check_format_string
            raise ValueError("Invalid format string")
        return _LazyCell(self._expression.dt.to_string(format=format_string))

    # ------------------------------------------------------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------------------------------------------------------

    def _equals(self, other: object) -> bool:
        if not isinstance(other, _LazyTemporalCell):
            return NotImplemented
        if self is other:
            return True
        return self._expression.meta.eq(other._expression.meta)


def _check_format_string(format_string: str) -> bool:
    valid_format_codes = {
        "F": "the standard",
        "a": "abbreviated weekday name",
        "A": "full weekday name",
        "w": "weekday as a decimal number",
        "d": "day of the month as a zero-padded decimal number",
        "b": "abbreviated month name",
        "B": "full month name",
        "m": "month as a zero-padded decimal number",
        "y": "year without century as a zero-padded decimal number",
        "Y": "year with century as a decimal number",
        "H": "hour (24-hour clock) as a zero-padded decimal number",
        "I": "hour (12-hour clock) as a zero-padded decimal number",
        "p": "locale's equivalent of either AM or PM",
        "M": "minute as a zero-padded decimal number",
        "S": "second as a zero-padded decimal number",
        "f": "microsecond as a zero-padded decimal number",
        "z": "UTC offset in the form ±HHMM[SS[.ffffff]]",
        "Z": "time zone name",
        "j": "day of the year as a zero-padded decimal number",
        "U": "week number of the year (Sunday as the first day of the week)",
        "W": "week number of the year (Monday as the first day of the week)",
        "c": "locale's appropriate date and time representation",
        "x": "locale's appropriate date representation",
        "X": "locale's appropriate time representation",
        "%": "a literal '%' character",
    }

    # Keep track of the positions in the string
    i = 0
    n = len(format_string)

    # Iterate over each character in the format string
    while i < n:
        if format_string[i] == "%":
            # Make sure there's at least one character following the '%'
            if i + 1 < n:
                code = format_string[i + 1]
                # Check if the following character is a valid format code
                if code not in valid_format_codes:
                    return False
                i += 2  # Skip ahead past the format code
            else:
                # '%' is at the end of the string with no following format code
                return False
        else:
            i += 1  # Continue to the next character

    return True
