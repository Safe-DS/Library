from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.data.tabular.containers._lazy_cell import _LazyCell

from ._datetime_operations import DatetimeOperations

if TYPE_CHECKING:
    import datetime as python_datetime

    import polars as pl

    from safeds.data.tabular.containers._cell import Cell


class _LazyDatetimeOperations(DatetimeOperations):
    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, expression: pl.Expr) -> None:
        self._expression: pl.Expr = expression

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _LazyDatetimeOperations):
            return NotImplemented
        if self is other:
            return True
        return self._expression.meta.eq(other._expression)

    def __hash__(self) -> int:
        return _structural_hash(self._expression.meta.serialize())

    def __repr__(self) -> str:
        return f"_LazyDatetimeOperations({self._expression})"

    def __sizeof__(self) -> int:
        return self._expression.__sizeof__()

    def __str__(self) -> str:
        return f"({self._expression}).dt"

    # ------------------------------------------------------------------------------------------------------------------
    # Extract components
    # ------------------------------------------------------------------------------------------------------------------

    def century(self) -> Cell[int | None]:
        return _LazyCell(self._expression.dt.century())

    def date(self) -> Cell[python_datetime.date | None]:
        return _LazyCell(self._expression.dt.date())

    def day(self) -> Cell[int | None]:
        return _LazyCell(self._expression.dt.day())

    def day_of_year(self) -> Cell[int | None]:
        return _LazyCell(self._expression.dt.ordinal_day())

    def hour(self) -> Cell[int | None]:
        return _LazyCell(self._expression.dt.hour())

    def microsecond(self) -> Cell[int | None]:
        return _LazyCell(self._expression.dt.microsecond())

    def millennium(self) -> Cell[int | None]:
        return _LazyCell(self._expression.dt.millennium())

    def millisecond(self) -> Cell[int | None]:
        return _LazyCell(self._expression.dt.millisecond())

    def minute(self) -> Cell[int | None]:
        return _LazyCell(self._expression.dt.minute())

    def month(self) -> Cell[int | None]:
        return _LazyCell(self._expression.dt.month())

    def quarter(self) -> Cell[int | None]:
        return _LazyCell(self._expression.dt.quarter())

    def second(self) -> Cell[int | None]:
        return _LazyCell(self._expression.dt.second())

    def time(self) -> Cell[python_datetime.time | None]:
        return _LazyCell(self._expression.dt.time())

    def week(self) -> Cell[int | None]:
        return _LazyCell(self._expression.dt.week())

    def weekday(self) -> Cell[int | None]:
        return _LazyCell(self._expression.dt.weekday())

    def year(self) -> Cell[int | None]:
        return _LazyCell(self._expression.dt.year())

    # ------------------------------------------------------------------------------------------------------------------
    # Other operations
    # ------------------------------------------------------------------------------------------------------------------

    def is_in_leap_year(self) -> Cell[bool | None]:
        return _LazyCell(self._expression.dt.is_leap_year())

    def to_string(self, *, format: str = "iso:strict") -> Cell[str | None]:
        if not _check_format_string(format):
            raise ValueError("Invalid format string")
        return _LazyCell(self._expression.dt.to_string(format=format))

    def unix_time(self) -> Cell[int | None]:
        return _LazyCell(self._expression.dt.epoch())


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
