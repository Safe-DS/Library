from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from safeds._utils import _structural_hash

from ._cell import Cell, _to_polars_expression

if TYPE_CHECKING:
    import polars as pl

    from safeds._typing import _ConvertibleToBooleanCell, _ConvertibleToCell
    from safeds.data.tabular.query import DatetimeOperations, DurationOperations, MathOperations, StringOperations
    from safeds.data.tabular.typing import ColumnType

T = TypeVar("T")


class _LazyCell(Cell[T]):
    """
    A single value in a table.

    This implementation only builds an expression that will be evaluated when needed.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, expression: pl.Expr) -> None:
        self._expression: pl.Expr = expression

    # "Boolean" operators (actually bitwise) -----------------------------------

    def __invert__(self) -> Cell[bool | None]:
        import polars as pl

        return _LazyCell(self._expression.cast(pl.Boolean).__invert__())

    def __and__(self, other: _ConvertibleToBooleanCell) -> Cell[bool | None]:
        other = _to_polars_expression(other)
        return _LazyCell(self._expression.__and__(other))

    def __rand__(self, other: _ConvertibleToBooleanCell) -> Cell[bool | None]:
        other = _to_polars_expression(other)
        return _LazyCell(self._expression.__rand__(other))

    def __or__(self, other: _ConvertibleToBooleanCell) -> Cell[bool | None]:
        other = _to_polars_expression(other)
        return _LazyCell(self._expression.__or__(other))

    def __ror__(self, other: _ConvertibleToBooleanCell) -> Cell[bool | None]:
        other = _to_polars_expression(other)
        return _LazyCell(self._expression.__ror__(other))

    def __xor__(self, other: _ConvertibleToBooleanCell) -> Cell[bool | None]:
        other = _to_polars_expression(other)
        return _LazyCell(self._expression.__xor__(other))

    def __rxor__(self, other: _ConvertibleToBooleanCell) -> Cell[bool | None]:
        other = _to_polars_expression(other)
        return _LazyCell(self._expression.__rxor__(other))

    # Comparison ---------------------------------------------------------------

    def __eq__(self, other: _ConvertibleToCell) -> Cell[bool | None]:  # type: ignore[override]
        other = _to_polars_expression(other)
        return _LazyCell(self._expression.__eq__(other))

    def __ge__(self, other: _ConvertibleToCell) -> Cell[bool | None]:
        other = _to_polars_expression(other)
        return _LazyCell(self._expression.__ge__(other))

    def __gt__(self, other: _ConvertibleToCell) -> Cell[bool | None]:
        other = _to_polars_expression(other)
        return _LazyCell(self._expression.__gt__(other))

    def __le__(self, other: _ConvertibleToCell) -> Cell[bool | None]:
        other = _to_polars_expression(other)
        return _LazyCell(self._expression.__le__(other))

    def __lt__(self, other: _ConvertibleToCell) -> Cell[bool | None]:
        other = _to_polars_expression(other)
        return _LazyCell(self._expression.__lt__(other))

    def __ne__(self, other: _ConvertibleToCell) -> Cell[bool | None]:  # type: ignore[override]
        other = _to_polars_expression(other)
        return _LazyCell(self._expression.__ne__(other))

    # Numeric operators --------------------------------------------------------

    def __abs__(self) -> Cell:
        return _LazyCell(self._expression.__abs__())

    def __ceil__(self) -> Cell:
        return _LazyCell(self._expression.ceil())

    def __floor__(self) -> Cell:
        return _LazyCell(self._expression.floor())

    def __neg__(self) -> Cell:
        return _LazyCell(self._expression.__neg__())

    def __pos__(self) -> Cell:
        return _LazyCell(self._expression.__pos__())

    def __add__(self, other: _ConvertibleToCell) -> Cell:
        other = _to_polars_expression(other)
        return _LazyCell(self._expression.__add__(other))

    def __radd__(self, other: _ConvertibleToCell) -> Cell:
        other = _to_polars_expression(other)
        return _LazyCell(self._expression.__radd__(other))

    def __floordiv__(self, other: _ConvertibleToCell) -> Cell:
        other = _to_polars_expression(other)
        return _LazyCell(self._expression.__floordiv__(other))

    def __rfloordiv__(self, other: _ConvertibleToCell) -> Cell:
        other = _to_polars_expression(other)
        return _LazyCell(self._expression.__rfloordiv__(other))

    def __mod__(self, other: _ConvertibleToCell) -> Cell:
        other = _to_polars_expression(other)
        return _LazyCell(self._expression.__mod__(other))

    def __rmod__(self, other: _ConvertibleToCell) -> Cell:
        other = _to_polars_expression(other)
        return _LazyCell(self._expression.__rmod__(other))

    def __mul__(self, other: _ConvertibleToCell) -> Cell:
        other = _to_polars_expression(other)
        return _LazyCell(self._expression.__mul__(other))

    def __rmul__(self, other: _ConvertibleToCell) -> Cell:
        other = _to_polars_expression(other)
        return _LazyCell(self._expression.__rmul__(other))

    def __pow__(self, other: _ConvertibleToCell) -> Cell:
        other = _to_polars_expression(other)
        return _LazyCell(self._expression.__pow__(other))

    def __rpow__(self, other: _ConvertibleToCell) -> Cell:
        other = _to_polars_expression(other)
        return _LazyCell(self._expression.__rpow__(other))

    def __sub__(self, other: _ConvertibleToCell) -> Cell:
        other = _to_polars_expression(other)
        return _LazyCell(self._expression.__sub__(other))

    def __rsub__(self, other: _ConvertibleToCell) -> Cell:
        other = _to_polars_expression(other)
        return _LazyCell(self._expression.__rsub__(other))

    def __truediv__(self, other: _ConvertibleToCell) -> Cell:
        other = _to_polars_expression(other)
        return _LazyCell(self._expression.__truediv__(other))

    def __rtruediv__(self, other: _ConvertibleToCell) -> Cell:
        other = _to_polars_expression(other)
        return _LazyCell(self._expression.__rtruediv__(other))

    # Other --------------------------------------------------------------------

    def __hash__(self) -> int:
        return _structural_hash(self._expression.meta.serialize())

    def __repr__(self) -> str:
        return f"_LazyCell({self._expression})"

    def __sizeof__(self) -> int:
        return self._expression.__sizeof__()

    def __str__(self) -> str:
        return self._expression.__str__()

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def dt(self) -> DatetimeOperations:
        from safeds.data.tabular.query._lazy_datetime_operations import _LazyDatetimeOperations  # circular import

        return _LazyDatetimeOperations(self._expression)

    @property
    def dur(self) -> DurationOperations:
        from safeds.data.tabular.query._lazy_duration_operations import _LazyDurationOperations  # circular import

        return _LazyDurationOperations(self._expression)

    @property
    def math(self) -> MathOperations:
        from safeds.data.tabular.query._lazy_math_operations import _LazyMathOperations  # circular import

        return _LazyMathOperations(self._expression)

    @property
    def str(self) -> StringOperations:
        from safeds.data.tabular.query._lazy_string_operations import _LazyStringOperations  # circular import

        return _LazyStringOperations(self._expression)

    # ------------------------------------------------------------------------------------------------------------------
    # Comparison operations
    # ------------------------------------------------------------------------------------------------------------------
    def eq(self, other: _ConvertibleToCell, *, propagate_missing_values: bool = True) -> Cell[bool | None]:
        other = _to_polars_expression(other)

        if propagate_missing_values:
            return _LazyCell(self._expression.eq(other))
        else:
            return _LazyCell(self._expression.eq_missing(other))

    def neq(self, other: _ConvertibleToCell, *, propagate_missing_values: bool = True) -> Cell[bool | None]:
        other = _to_polars_expression(other)

        if propagate_missing_values:
            return _LazyCell(self._expression.ne(other))
        else:
            return _LazyCell(self._expression.ne_missing(other))

    # ------------------------------------------------------------------------------------------------------------------
    # Other
    # ------------------------------------------------------------------------------------------------------------------

    def cast(self, type: ColumnType) -> Cell:
        return _LazyCell(self._expression.cast(type._polars_data_type))

    # ------------------------------------------------------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def _polars_expression(self) -> pl.Expr:
        return self._expression

    def _equals(self, other: object) -> bool:
        if not isinstance(other, _LazyCell):
            return NotImplemented
        if self is other:
            return True
        return self._expression.meta.eq(other._expression)
