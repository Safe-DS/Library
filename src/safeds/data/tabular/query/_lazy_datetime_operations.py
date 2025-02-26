from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from safeds._utils import _structural_hash
from safeds._validation import _convert_and_check_datetime_format
from safeds.data.tabular.containers._cell import _to_polars_expression
from safeds.data.tabular.containers._lazy_cell import _LazyCell

from ._datetime_operations import DatetimeOperations

if TYPE_CHECKING:
    import datetime as python_datetime

    import polars as pl

    from safeds._typing import _ConvertibleToIntCell
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

    def day_of_week(self) -> Cell[int | None]:
        return _LazyCell(self._expression.dt.weekday())

    def year(self) -> Cell[int | None]:
        return _LazyCell(self._expression.dt.year())

    # ------------------------------------------------------------------------------------------------------------------
    # Other operations
    # ------------------------------------------------------------------------------------------------------------------

    def is_in_leap_year(self) -> Cell[bool | None]:
        return _LazyCell(self._expression.dt.is_leap_year())

    def replace(
        self,
        *,
        year: _ConvertibleToIntCell = None,
        month: _ConvertibleToIntCell = None,
        day: _ConvertibleToIntCell = None,
        hour: _ConvertibleToIntCell = None,
        minute: _ConvertibleToIntCell = None,
        second: _ConvertibleToIntCell = None,
        microsecond: _ConvertibleToIntCell = None,
    ) -> Cell:
        year = _to_polars_expression(year)
        month = _to_polars_expression(month)
        day = _to_polars_expression(day)
        hour = _to_polars_expression(hour)
        minute = _to_polars_expression(minute)
        second = _to_polars_expression(second)
        microsecond = _to_polars_expression(microsecond)

        return _LazyCell(
            self._expression.dt.replace(
                year=year,
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                second=second,
                microsecond=microsecond,
            ),
        )

    def to_string(self, *, format: str = "iso") -> Cell[str | None]:
        if format == "iso":
            format = "iso:strict"  # noqa: A001
        else:
            format = _convert_and_check_datetime_format(format, type_="datetime", used_for_parsing=False)  # noqa: A001

        return _LazyCell(self._expression.dt.to_string(format=format))

    def unix_timestamp(self, *, unit: Literal["s", "ms", "us"] = "s") -> Cell[int | None]:
        return _LazyCell(self._expression.dt.epoch(time_unit=unit))
