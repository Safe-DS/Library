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

    def arccos(self) -> Cell:
        return _LazyCell(self._expression.arccos())

    def arccosh(self) -> Cell:
        return _LazyCell(self._expression.arccosh())

    def arcsin(self) -> Cell:
        return _LazyCell(self._expression.arcsin())

    def arcsinh(self) -> Cell:
        return _LazyCell(self._expression.arcsinh())

    def arctan(self) -> Cell:
        return _LazyCell(self._expression.arctan())

    def arctanh(self) -> Cell:
        return _LazyCell(self._expression.arctanh())

    def ceil(self) -> Cell:
        return _LazyCell(self._expression.ceil())

    def cos(self) -> Cell:
        return _LazyCell(self._expression.cos())

    def cosh(self) -> Cell:
        return _LazyCell(self._expression.cosh())

    def floor(self) -> Cell:
        return _LazyCell(self._expression.floor())

    def round_to_decimal_places(self, decimal_places: int) -> Cell:
        return _LazyCell(self._expression.round(decimal_places))

    def round_to_significant_figures(self, significant_figures: int) -> Cell:
        return _LazyCell(self._expression.round_sig_figs(significant_figures))

    def sign(self) -> Cell:
        return _LazyCell(self._expression.sign())

    def sin(self) -> Cell:
        return _LazyCell(self._expression.sin())

    def sinh(self) -> Cell:
        return _LazyCell(self._expression.sinh())

    def tan(self) -> Cell:
        return _LazyCell(self._expression.tan())

    def tanh(self) -> Cell:
        return _LazyCell(self._expression.tanh())
