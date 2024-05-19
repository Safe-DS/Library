from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _ClosedBound

from ._lazy_cell import _LazyCell
from ._string_cell import StringCell

if TYPE_CHECKING:
    import datetime

    import polars as pl

    from ._cell import Cell


class _LazyStringCell(StringCell):
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
    # String operations
    # ------------------------------------------------------------------------------------------------------------------

    def contains(self, substring: str) -> Cell[bool]:
        return _LazyCell(self._expression.str.contains(substring, literal=True))

    def length(self, optimize_for_ascii: bool = False) -> Cell[int]:
        if optimize_for_ascii:
            return _LazyCell(self._expression.str.len_bytes())
        else:
            return _LazyCell(self._expression.str.len_chars())

    def ends_with(self, suffix: str) -> Cell[bool]:
        return _LazyCell(self._expression.str.ends_with(suffix))

    def index_of(self, substring: str) -> Cell[int | None]:
        return _LazyCell(self._expression.str.find(substring, literal=True))

    def replace(self, old: str, new: str) -> Cell[str]:
        return _LazyCell(self._expression.str.replace_all(old, new, literal=True))

    def starts_with(self, prefix: str) -> Cell[bool]:
        return _LazyCell(self._expression.str.starts_with(prefix))

    def substring(self, start: int = 0, length: int | None = None) -> Cell[str]:
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

    def to_lowercase(self) -> Cell[str]:
        return _LazyCell(self._expression.str.to_lowercase())

    def to_uppercase(self) -> Cell[str]:
        return _LazyCell(self._expression.str.to_uppercase())

    def trim(self) -> Cell[str]:
        return _LazyCell(self._expression.str.strip_chars())

    def trim_end(self) -> Cell[str]:
        return _LazyCell(self._expression.str.strip_chars_end())

    def trim_start(self) -> Cell[str]:
        return _LazyCell(self._expression.str.strip_chars_start())

    # ------------------------------------------------------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------------------------------------------------------

    def _equals(self, other: object) -> bool:
        if not isinstance(other, _LazyStringCell):
            return NotImplemented
        if self is other:
            return True
        return self._expression.meta.eq(other._expression.meta)
