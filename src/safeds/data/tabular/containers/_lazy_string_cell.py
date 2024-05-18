from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash

from ._lazy_cell import _LazyCell
from ._string_cell import StringCell

if TYPE_CHECKING:
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

    def starts_with(self, prefix: str) -> Cell[bool]:
        return _LazyCell(self._expression.str.starts_with(prefix))

    def to_lowercase(self) -> Cell[str]:
        return _LazyCell(self._expression.str.to_lowercase())

    def to_uppercase(self) -> Cell[str]:
        return _LazyCell(self._expression.str.to_uppercase())

    # ------------------------------------------------------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------------------------------------------------------

    def _equals(self, other: object) -> bool:
        if not isinstance(other, _LazyStringCell):
            return NotImplemented
        if self is other:
            return True
        return self._expression.meta.eq(other._expression.meta)
