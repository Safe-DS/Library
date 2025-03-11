from __future__ import annotations
from typing import TYPE_CHECKING


from src.safeds._utils import _structural_hash

from src.safeds.data.tabular.query._column_selector import ColumnSelector


if TYPE_CHECKING:
    import polars as pl


class _LazyColumnSelector(ColumnSelector):
    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, expression: pl.Expr) -> None:
        self._expression: pl.Expr = expression

    def __invert__(self) -> ColumnSelector:
        return _LazyColumnSelector(self._expression.__invert__())

    def __and__(self, other: ColumnSelector) -> ColumnSelector:
        return _LazyColumnSelector(self._expression.__and__(other._polars_expression))

    def __rand__(self, other: ColumnSelector) -> ColumnSelector:
        return _LazyColumnSelector(self._expression.__rand__(other._polars_expression))

    def __or__(self, other: ColumnSelector) -> ColumnSelector:
        return _LazyColumnSelector(self._expression.__or__(other._polars_expression))

    def __ror__(self, other: ColumnSelector) -> ColumnSelector:
        return _LazyColumnSelector(self._expression.__ror__(other._polars_expression))

    def __sub__(self, other: ColumnSelector) -> ColumnSelector:
        return _LazyColumnSelector(self._expression.__sub__(other._polars_expression))

    def __rsub__(self, other: ColumnSelector) -> ColumnSelector:
        return _LazyColumnSelector(self._expression.__rsub__(other._polars_expression))

    def __xor__(self, other: ColumnSelector) -> ColumnSelector:
        return _LazyColumnSelector(self._expression.__xor__(other._polars_expression))

    def __rxor__(self, other: ColumnSelector) -> ColumnSelector:
        return _LazyColumnSelector(self._expression.__rxor__(other._polars_expression))

    # Other --------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _LazyColumnSelector):
            return NotImplemented
        if self is other:
            return True
        return self._expression.meta.eq(other._expression)

    def __hash__(self) -> int:
        return _structural_hash(self._expression.meta.serialize())

    def __repr__(self) -> str:
        return f"_LazyColumnSelector({self._expression})"

    def __sizeof__(self) -> int:
        return self._expression.__sizeof__()

    def __str__(self) -> str:
        return self._expression.__str__()

    # ------------------------------------------------------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def _polars_expression(self) -> pl.Expr:
        return self._expression
