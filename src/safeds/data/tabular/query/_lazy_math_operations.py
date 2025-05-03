from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _ClosedBound, _OpenBound
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

    def acos(self) -> Cell:
        return _LazyCell(self._expression.arccos())

    def acosh(self) -> Cell:
        return _LazyCell(self._expression.arccosh())

    def asin(self) -> Cell:
        return _LazyCell(self._expression.arcsin())

    def asinh(self) -> Cell:
        return _LazyCell(self._expression.arcsinh())

    def atan(self) -> Cell:
        return _LazyCell(self._expression.arctan())

    def atanh(self) -> Cell:
        return _LazyCell(self._expression.arctanh())

    def cbrt(self) -> Cell:
        return _LazyCell(self._expression.cbrt())

    def ceil(self) -> Cell:
        return _LazyCell(self._expression.ceil())

    def cos(self) -> Cell:
        return _LazyCell(self._expression.cos())

    def cosh(self) -> Cell:
        return _LazyCell(self._expression.cosh())

    def degrees_to_radians(self) -> Cell:
        return _LazyCell(self._expression.radians())

    def exp(self) -> Cell:
        return _LazyCell(self._expression.exp())

    def floor(self) -> Cell:
        return _LazyCell(self._expression.floor())

    def ln(self) -> Cell:
        return _LazyCell(self._expression.log())

    def log(self, base: float) -> Cell:
        _check_bounds("base", base, lower_bound=_OpenBound(0))
        if base == 1:
            raise ValueError("The base of the logarithm must not be 1.")

        return _LazyCell(self._expression.log(base))

    def log10(self) -> Cell:
        return _LazyCell(self._expression.log10())

    def radians_to_degrees(self) -> Cell:
        return _LazyCell(self._expression.degrees())

    def round_to_decimal_places(self, decimal_places: int) -> Cell:
        _check_bounds("decimal_places", decimal_places, lower_bound=_ClosedBound(0))

        return _LazyCell(self._expression.round(decimal_places, mode="half_away_from_zero"))

    def round_to_significant_figures(self, significant_figures: int) -> Cell:
        _check_bounds("significant_figures", significant_figures, lower_bound=_ClosedBound(1))

        return _LazyCell(self._expression.round_sig_figs(significant_figures))

    def sign(self) -> Cell:
        return _LazyCell(self._expression.sign())

    def sin(self) -> Cell:
        return _LazyCell(self._expression.sin())

    def sinh(self) -> Cell:
        return _LazyCell(self._expression.sinh())

    def sqrt(self) -> Cell:
        return _LazyCell(self._expression.sqrt())

    def tan(self) -> Cell:
        return _LazyCell(self._expression.tan())

    def tanh(self) -> Cell:
        return _LazyCell(self._expression.tanh())
