from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _ClosedBound, _convert_and_check_datetime_format
from safeds.data.tabular.containers._cell import _to_polars_expression
from safeds.data.tabular.containers._lazy_cell import _LazyCell
from safeds.data.tabular.typing import ColumnType

from ._string_operations import StringOperations

if TYPE_CHECKING:
    import datetime

    import polars as pl

    from safeds._typing import _ConvertibleToIntCell, _ConvertibleToStringCell
    from safeds.data.tabular.containers._cell import Cell


class _LazyStringOperations(StringOperations):
    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, expression: pl.Expr) -> None:
        self._expression: pl.Expr = expression

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _LazyStringOperations):
            return NotImplemented
        if self is other:
            return True
        return self._expression.meta.eq(other._expression)

    def __hash__(self) -> int:
        return _structural_hash(self._expression.meta.serialize())

    def __repr__(self) -> str:
        return f"_LazyStringOperations({self._expression})"

    def __sizeof__(self) -> int:
        return self._expression.__sizeof__()

    def __str__(self) -> str:
        return f"({self._expression}).str"

    # ------------------------------------------------------------------------------------------------------------------
    # String operations
    # ------------------------------------------------------------------------------------------------------------------

    def contains(self, substring: _ConvertibleToStringCell) -> Cell[bool | None]:
        substring = _to_polars_expression(substring, type_if_none=ColumnType.string())

        return _LazyCell(self._expression.str.contains(substring, literal=True))

    def ends_with(self, suffix: _ConvertibleToStringCell) -> Cell[bool | None]:
        suffix = _to_polars_expression(suffix)

        return _LazyCell(self._expression.str.ends_with(suffix))

    def index_of(self, substring: _ConvertibleToStringCell) -> Cell[int | None]:
        substring = _to_polars_expression(substring, type_if_none=ColumnType.string())

        return _LazyCell(self._expression.str.find(substring, literal=True))

    def length(self, optimize_for_ascii: bool = False) -> Cell[int | None]:
        if optimize_for_ascii:
            return _LazyCell(self._expression.str.len_bytes())
        else:
            return _LazyCell(self._expression.str.len_chars())

    def pad_end(self, length: int, *, character: str = " ") -> Cell[str | None]:
        _check_bounds("length", length, lower_bound=_ClosedBound(0))
        if len(character) != 1:
            raise ValueError("Can only pad with a single character.")

        return _LazyCell(self._expression.str.pad_end(length, character))

    def pad_start(self, length: int, *, character: str = " ") -> Cell[str | None]:
        _check_bounds("length", length, lower_bound=_ClosedBound(0))
        if len(character) != 1:
            raise ValueError("Can only pad with a single character.")

        return _LazyCell(self._expression.str.pad_start(length, character))

    def repeat(self, count: _ConvertibleToIntCell) -> Cell[str | None]:
        if isinstance(count, int):
            _check_bounds("count", count, lower_bound=_ClosedBound(0))

        count = _to_polars_expression(count)

        return _LazyCell(self._expression.repeat_by(count).list.join("", ignore_nulls=False))

    def remove_prefix(self, prefix: _ConvertibleToStringCell) -> Cell[str | None]:
        prefix = _to_polars_expression(prefix, type_if_none=ColumnType.string())

        return _LazyCell(self._expression.str.strip_prefix(prefix))

    def remove_suffix(self, suffix: _ConvertibleToStringCell) -> Cell[str | None]:
        suffix = _to_polars_expression(suffix, type_if_none=ColumnType.string())

        return _LazyCell(self._expression.str.strip_suffix(suffix))

    def replace_all(self, old: _ConvertibleToStringCell, new: _ConvertibleToStringCell) -> Cell[str | None]:
        old = _to_polars_expression(old, type_if_none=ColumnType.string())
        new = _to_polars_expression(new, type_if_none=ColumnType.string())

        return _LazyCell(self._expression.str.replace_all(old, new, literal=True))

    def reverse(self) -> Cell[str | None]:
        return _LazyCell(self._expression.str.reverse())

    def slice(
        self,
        *,
        start: _ConvertibleToIntCell = 0,
        length: _ConvertibleToIntCell = None,
    ) -> Cell[str | None]:
        if isinstance(length, int):
            _check_bounds("length", length, lower_bound=_ClosedBound(0))

        start = _to_polars_expression(start)
        length = _to_polars_expression(length)

        return _LazyCell(self._expression.str.slice(start, length))

    def starts_with(self, prefix: _ConvertibleToStringCell) -> Cell[bool | None]:
        prefix = _to_polars_expression(prefix)

        return _LazyCell(self._expression.str.starts_with(prefix))

    def strip(self, *, characters: _ConvertibleToStringCell = None) -> Cell[str | None]:
        characters = _to_polars_expression(characters)

        return _LazyCell(self._expression.str.strip_chars(characters))

    def strip_end(self, *, characters: _ConvertibleToStringCell = None) -> Cell[str | None]:
        characters = _to_polars_expression(characters)

        return _LazyCell(self._expression.str.strip_chars_end(characters))

    def strip_start(self, *, characters: _ConvertibleToStringCell = None) -> Cell[str | None]:
        characters = _to_polars_expression(characters)

        return _LazyCell(self._expression.str.strip_chars_start(characters))

    def to_date(self, *, format: str | None = "iso") -> Cell[datetime.date | None]:
        if format == "iso":
            format = "%F"  # noqa: A001
        elif format is not None:
            format = _convert_and_check_datetime_format(format, type_="date", used_for_parsing=True)  # noqa: A001

        return _LazyCell(self._expression.str.to_date(format=format, strict=False))

    def to_datetime(self, *, format: str | None = "iso") -> Cell[datetime.datetime | None]:
        if format == "iso":
            format = "%+"  # noqa: A001
        elif format is not None:
            format = _convert_and_check_datetime_format(format, type_="datetime", used_for_parsing=True)  # noqa: A001

        return _LazyCell(self._expression.str.to_datetime(format=format, strict=False))

    def to_float(self) -> Cell[float | None]:
        import polars as pl

        return _LazyCell(self._expression.cast(pl.Float64(), strict=False))

    def to_int(self, *, base: _ConvertibleToIntCell = 10) -> Cell[int | None]:
        base = _to_polars_expression(base)

        return _LazyCell(self._expression.str.to_integer(base=base, strict=False))

    def to_lowercase(self) -> Cell[str | None]:
        return _LazyCell(self._expression.str.to_lowercase())

    def to_time(self, *, format: str | None = "iso") -> Cell[datetime.time | None]:
        if format == "iso":
            format = "%T%.f"  # noqa: A001
        elif format is not None:
            format = _convert_and_check_datetime_format(format, type_="time", used_for_parsing=True)  # noqa: A001

        return _LazyCell(self._expression.str.to_time(format=format, strict=False))

    def to_uppercase(self) -> Cell[str | None]:
        return _LazyCell(self._expression.str.to_uppercase())
