from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.data.tabular.containers._lazy_cell import _LazyCell
from safeds.data.tabular.query._math_operations import MathOperations

if TYPE_CHECKING:
    import polars as pl

    from safeds.data.tabular.containers import Cell


class _LazyMathOperations(MathOperations):
    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, expression: pl.Expr) -> None:
        self._expression: pl.Expr = expression

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _LazyMathOperations):
            return NotImplemented
        if self is other:
            return True
        return self._expression.meta.eq(other._expression)

    def __hash__(self) -> int:
        return _structural_hash(self._expression.meta.serialize())

    def __repr__(self) -> str:
        return f"_LazyMathOperations({self._expression})"

    def __sizeof__(self) -> int:
        return self._expression.__sizeof__()

    def __str__(self) -> str:
        return f"({self._expression}).math"

    # ------------------------------------------------------------------------------------------------------------------
    # Math operations
    # ------------------------------------------------------------------------------------------------------------------

    def abs(self) -> Cell:
        return _LazyCell(self._expression.__abs__())

    def ceil(self) -> Cell:
        return _LazyCell(self._expression.ceil())

    def floor(self) -> Cell:
        return _LazyCell(self._expression.floor())
