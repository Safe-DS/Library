from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from safeds._utils import _structural_hash
from safeds.data.tabular.typing._experimental_polars_data_type import _PolarsDataType

from ._experimental_cell import ExperimentalCell

if TYPE_CHECKING:
    import polars as pl

    from safeds.data.tabular.typing._experimental_data_type import ExperimentalDataType

    from ._experimental_column import ExperimentalColumn

T = TypeVar("T")
P = TypeVar("P")
R = TypeVar("R")


class _VectorizedCell(ExperimentalCell[T]):
    """
    A single value in a table.

    This implementation treats an entire column as a cell. This greatly speeds up operations on the cell.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _from_polars_series(data: pl.Series) -> _VectorizedCell:
        result = object.__new__(_VectorizedCell)
        result._series = data
        return result

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, column: ExperimentalColumn[T]) -> None:
        self._series: pl.Series = column._series

    # "Boolean" operators (actually bitwise) -----------------------------------

    def __invert__(self) -> ExperimentalCell[bool]:
        import polars as pl

        if self._series.dtype != pl.Boolean:
            return NotImplemented

        return _wrap(self._series.__invert__())

    def __and__(self, other: bool | ExperimentalCell[bool]) -> ExperimentalCell[bool]:
        right_operand = _normalize_boolean_operation_operands(self, other)
        if right_operand is None:
            return NotImplemented

        return _wrap(self._series.__and__(right_operand))

    def __rand__(self, other: bool | ExperimentalCell[bool]) -> ExperimentalCell[bool]:
        right_operand = _normalize_boolean_operation_operands(self, other)
        if right_operand is None:
            return NotImplemented

        return _wrap(self._series.__rand__(right_operand))

    def __or__(self, other: bool | ExperimentalCell[bool]) -> ExperimentalCell[bool]:
        right_operand = _normalize_boolean_operation_operands(self, other)
        if right_operand is None:
            return NotImplemented

        return _wrap(self._series.__or__(right_operand))

    def __ror__(self, other: bool | ExperimentalCell[bool]) -> ExperimentalCell[bool]:
        right_operand = _normalize_boolean_operation_operands(self, other)
        if right_operand is None:
            return NotImplemented

        return _wrap(self._series.__ror__(right_operand))

    def __xor__(self, other: bool | ExperimentalCell[bool]) -> ExperimentalCell[bool]:
        right_operand = _normalize_boolean_operation_operands(self, other)
        if right_operand is None:
            return NotImplemented

        return _wrap(self._series.__xor__(right_operand))

    def __rxor__(self, other: bool | ExperimentalCell[bool]) -> ExperimentalCell[bool]:
        right_operand = _normalize_boolean_operation_operands(self, other)
        if right_operand is None:
            return NotImplemented

        return _wrap(self._series.__rxor__(right_operand))

    # Comparison ---------------------------------------------------------------

    def __eq__(self, other: object) -> ExperimentalCell[bool]:  # type: ignore[override]
        other = _unwrap(other)
        return _wrap(self._series.__eq__(other))

    def __ge__(self, other: Any) -> ExperimentalCell[bool]:
        other = _unwrap(other)
        return _wrap(self._series.__ge__(other))

    def __gt__(self, other: Any) -> ExperimentalCell[bool]:
        other = _unwrap(other)
        return _wrap(self._series.__gt__(other))

    def __le__(self, other: Any) -> ExperimentalCell[bool]:
        other = _unwrap(other)
        return _wrap(self._series.__le__(other))

    def __lt__(self, other: Any) -> ExperimentalCell[bool]:
        other = _unwrap(other)
        return _wrap(self._series.__lt__(other))

    def __ne__(self, other: object) -> ExperimentalCell[bool]:  # type: ignore[override]
        other = _unwrap(other)
        return _wrap(self._series.__ne__(other))

    # Numeric operators --------------------------------------------------------

    def __abs__(self) -> ExperimentalCell[R]:
        return _wrap(self._series.__abs__())

    def __ceil__(self) -> ExperimentalCell[R]:
        return _wrap(self._series.ceil())

    def __floor__(self) -> ExperimentalCell[R]:
        return _wrap(self._series.floor())

    def __neg__(self) -> ExperimentalCell[R]:
        return _wrap(self._series.__neg__())

    def __pos__(self) -> ExperimentalCell[R]:
        return _wrap(self._series.__pos__())

    def __add__(self, other: Any) -> ExperimentalCell[R]:
        other = _unwrap(other)
        return _wrap(self._series.__add__(other))

    def __radd__(self, other: Any) -> ExperimentalCell[R]:
        other = _unwrap(other)
        return _wrap(self._series.__radd__(other))

    def __floordiv__(self, other: Any) -> ExperimentalCell[R]:
        other = _unwrap(other)
        return _wrap(self._series.__floordiv__(other))

    def __rfloordiv__(self, other: Any) -> ExperimentalCell[R]:
        other = _unwrap(other)
        return _wrap(self._series.__rfloordiv__(other))

    def __mod__(self, other: Any) -> ExperimentalCell[R]:
        other = _unwrap(other)
        return _wrap(self._series.__mod__(other))

    def __rmod__(self, other: Any) -> ExperimentalCell[R]:
        other = _unwrap(other)
        return _wrap(self._series.__rmod__(other))

    def __mul__(self, other: Any) -> ExperimentalCell[R]:
        other = _unwrap(other)
        return _wrap(self._series.__mul__(other))

    def __rmul__(self, other: Any) -> ExperimentalCell[R]:
        other = _unwrap(other)
        return _wrap(self._series.__rmul__(other))

    def __pow__(self, other: float | ExperimentalCell[P]) -> ExperimentalCell[R]:
        other = _unwrap(other)
        return _wrap(self._series.__pow__(other))

    def __rpow__(self, other: float | ExperimentalCell[P]) -> ExperimentalCell[R]:
        other = _unwrap(other)
        return _wrap(self._series.__rpow__(other))

    def __sub__(self, other: Any) -> ExperimentalCell[R]:
        other = _unwrap(other)
        return _wrap(self._series.__sub__(other))

    def __rsub__(self, other: Any) -> ExperimentalCell[R]:
        other = _unwrap(other)
        return _wrap(self._series.__rsub__(other))

    def __truediv__(self, other: Any) -> ExperimentalCell[R]:
        other = _unwrap(other)
        return _wrap(self._series.__truediv__(other))

    def __rtruediv__(self, other: Any) -> ExperimentalCell[R]:
        other = _unwrap(other)
        return _wrap(self._series.__rtruediv__(other))

    # String representation ----------------------------------------------------

    def __repr__(self) -> str:
        return self._series.__repr__()

    def __str__(self) -> str:
        return self._series.__str__()

    # Other --------------------------------------------------------------------

    def __hash__(self) -> int:
        return _structural_hash(
            self._series.name,
            self.type.__repr__(),
            self._series.len(),
        )

    def __sizeof__(self) -> int:
        return self._series.estimated_size()

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def type(self) -> ExperimentalDataType:
        return _PolarsDataType(self._series.dtype)

    # ------------------------------------------------------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def _polars_expression(self) -> pl.Series:
        return self._series

    def _equals(self, other: object) -> bool:
        if not isinstance(other, _VectorizedCell):
            return NotImplemented
        if self is other:
            return True
        return self._series.equals(other._series)


def _normalize_boolean_operation_operands(
    left_operand: _VectorizedCell,
    right_operand: bool | ExperimentalCell[bool],
) -> pl.Series | None:
    """
    Normalize the operands of a boolean operation (not, and, or, xor).

    If one of the operands is invalid, None is returned. Otherwise, the normalized right operand is returned.
    """
    import polars as pl

    if left_operand._series.dtype != pl.Boolean:
        return None
    elif isinstance(right_operand, bool):
        return pl.Series("", [right_operand])
    elif not isinstance(right_operand, _VectorizedCell) or right_operand._series.dtype != pl.Boolean:
        return None
    else:
        return right_operand._series


def _wrap(other: pl.Series) -> Any:
    return _VectorizedCell._from_polars_series(other)


def _unwrap(other: Any) -> Any:
    if isinstance(other, _VectorizedCell):
        return other._series
    return other
