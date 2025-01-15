from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _ClosedBound
from safeds.data.tabular.containers._lazy_cell import _LazyCell

from ._string_operations import StringOperations

if TYPE_CHECKING:
    import datetime

    import polars as pl

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

    def contains(self, substring: str) -> Cell[bool | None]:
        return _LazyCell(self._expression.str.contains(substring, literal=True))

    def length(self, optimize_for_ascii: bool = False) -> Cell[int | None]:
        if optimize_for_ascii:
            return _LazyCell(self._expression.str.len_bytes())
        else:
            return _LazyCell(self._expression.str.len_chars())

    def ends_with(self, suffix: str) -> Cell[bool | None]:
        return _LazyCell(self._expression.str.ends_with(suffix))

    def index_of(self, substring: str) -> Cell[int | None]:
        return _LazyCell(self._expression.str.find(substring, literal=True))

    def replace(self, old: str, new: str) -> Cell[str | None]:
        return _LazyCell(self._expression.str.replace_all(old, new, literal=True))

    def starts_with(self, prefix: str) -> Cell[bool | None]:
        return _LazyCell(self._expression.str.starts_with(prefix))

    def substring(self, start: int = 0, length: int | None = None) -> Cell[str | None]:
        _check_bounds("length", length, lower_bound=_ClosedBound(0))

        return _LazyCell(self._expression.str.slice(start, length))

    def to_date(self) -> Cell[datetime.date | None]:
        return _LazyCell(self._expression.str.to_date(format="%F", strict=False))

    def to_datetime(self) -> Cell[datetime.datetime | None]:
        return _LazyCell(self._expression.str.to_datetime(format="%+", strict=False))

    def to_int(self, *, base: int = 10) -> Cell[int | None]:
        return _LazyCell(self._expression.str.to_integer(base=base, strict=False))

    def to_float(self) -> Cell[float | None]:
        import polars as pl

        return _LazyCell(self._expression.cast(pl.Float64, strict=False))

    def to_lowercase(self) -> Cell[str | None]:
        return _LazyCell(self._expression.str.to_lowercase())

    def to_uppercase(self) -> Cell[str | None]:
        return _LazyCell(self._expression.str.to_uppercase())

    def trim(self) -> Cell[str | None]:
        return _LazyCell(self._expression.str.strip_chars())

    def trim_end(self) -> Cell[str | None]:
        return _LazyCell(self._expression.str.strip_chars_end())

    def trim_start(self) -> Cell[str | None]:
        return _LazyCell(self._expression.str.strip_chars_start())
