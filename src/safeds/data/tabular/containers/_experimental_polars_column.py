from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeVar, overload

from ._experimental_polars_cell import ExperimentalPolarsCell

if TYPE_CHECKING:
    from polars import Series

T = TypeVar("T")
P = TypeVar("P")
R = TypeVar("R")


class ExperimentalPolarsColumn(ExperimentalPolarsCell[T], Sequence[T]):
    """
    A column is a named, one-dimensional collection of homogeneous values.

    Parameters
    ----------
    name:
        The name of the column.
    data:
        The data. If None, an empty column is created.

    Examples
    --------
    >>> from safeds.data.tabular.containers import Column
    >>> column = Column("test", [1, 2, 3])
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _from_polars_series(data: Series) -> ExperimentalPolarsColumn:
        result = object.__new__(ExperimentalPolarsColumn)
        result._series = data
        return result

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, name: str, data: Sequence[T] | None = None) -> None:
        import polars as pl

        if data is None:
            data = []

        self._series: pl.Series = pl.Series(name, data)

    # "Boolean" operators (actually bitwise) -----------------------------------

    def __invert__(self) -> ExperimentalPolarsCell[bool]:
        import polars as pl

        if self._series.dtype != pl.Boolean:
            return NotImplemented

        return _wrap(self._series.__invert__())

    def __and__(self, other: bool | ExperimentalPolarsCell[bool]) -> ExperimentalPolarsCell[bool]:
        right_operand = _normalize_boolean_operation_operands(self, other)
        if right_operand is None:
            return NotImplemented

        return _wrap(self._series.__and__(right_operand))

    def __rand__(self, other: bool | ExperimentalPolarsCell[bool]) -> ExperimentalPolarsCell[bool]:
        right_operand = _normalize_boolean_operation_operands(self, other)
        if right_operand is None:
            return NotImplemented

        return _wrap(self._series.__rand__(right_operand))

    def __or__(self, other: bool | ExperimentalPolarsCell[bool]) -> ExperimentalPolarsCell[bool]:
        right_operand = _normalize_boolean_operation_operands(self, other)
        if right_operand is None:
            return NotImplemented

        return _wrap(self._series.__or__(right_operand))

    def __ror__(self, other: bool | ExperimentalPolarsCell[bool]) -> ExperimentalPolarsCell[bool]:
        right_operand = _normalize_boolean_operation_operands(self, other)
        if right_operand is None:
            return NotImplemented

        return _wrap(self._series.__ror__(right_operand))

    def __xor__(self, other: bool | ExperimentalPolarsCell[bool]) -> ExperimentalPolarsCell[bool]:
        right_operand = _normalize_boolean_operation_operands(self, other)
        if right_operand is None:
            return NotImplemented

        return _wrap(self._series.__xor__(right_operand))

    def __rxor__(self, other: bool | ExperimentalPolarsCell[bool]) -> ExperimentalPolarsCell[bool]:
        right_operand = _normalize_boolean_operation_operands(self, other)
        if right_operand is None:
            return NotImplemented

        return _wrap(self._series.__rxor__(right_operand))

    # Collection ---------------------------------------------------------------

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> ExperimentalPolarsColumn[T]: ...

    def __getitem__(self, index: int | slice) -> T | ExperimentalPolarsColumn[T]:
        return self._series.__getitem__(index)

    def __len__(self) -> int:
        return self._series.__len__()

    # Comparison ---------------------------------------------------------------

    def __eq__(self, other: object) -> ExperimentalPolarsCell[bool]:  # type: ignore[override]
        other = _unwrap(other)
        return _wrap(self._series.__eq__(other))

    def __ge__(self, other: Any) -> ExperimentalPolarsCell[bool]:
        other = _unwrap(other)
        return _wrap(self._series.__ge__(other))

    def __gt__(self, other: Any) -> ExperimentalPolarsCell[bool]:
        other = _unwrap(other)
        return _wrap(self._series.__gt__(other))

    def __le__(self, other: Any) -> ExperimentalPolarsCell[bool]:
        other = _unwrap(other)
        return _wrap(self._series.__le__(other))

    def __lt__(self, other: Any) -> ExperimentalPolarsCell[bool]:
        other = _unwrap(other)
        return _wrap(self._series.__lt__(other))

    def __ne__(self, other: object) -> ExperimentalPolarsCell[bool]:  # type: ignore[override]
        other = _unwrap(other)
        return _wrap(self._series.__ne__(other))

    # Numeric operators --------------------------------------------------------

    def __abs__(self) -> ExperimentalPolarsCell[R]:
        return _wrap(self._series.__abs__())

    def __neg__(self) -> ExperimentalPolarsCell[R]:
        return _wrap(self._series.__neg__())

    def __pos__(self) -> ExperimentalPolarsCell[R]:
        return _wrap(self._series.__pos__())

    def __add__(self, other: Any) -> ExperimentalPolarsCell[R]:
        other = _unwrap(other)
        return _wrap(self._series.__add__(other))

    def __radd__(self, other: Any) -> ExperimentalPolarsCell[R]:
        other = _unwrap(other)
        return _wrap(self._series.__radd__(other))

    def __floordiv__(self, other: Any) -> ExperimentalPolarsCell[R]:
        other = _unwrap(other)
        return _wrap(self._series.__floordiv__(other))

    def __rfloordiv__(self, other: Any) -> ExperimentalPolarsCell[R]:
        other = _unwrap(other)
        return _wrap(self._series.__rfloordiv__(other))

    def __mod__(self, other: Any) -> ExperimentalPolarsCell[R]:
        other = _unwrap(other)
        return _wrap(self._series.__mod__(other))

    def __rmod__(self, other: Any) -> ExperimentalPolarsCell[R]:
        other = _unwrap(other)
        return _wrap(self._series.__rmod__(other))

    def __mul__(self, other: Any) -> ExperimentalPolarsCell[R]:
        other = _unwrap(other)
        return _wrap(self._series.__mul__(other))

    def __rmul__(self, other: Any) -> ExperimentalPolarsCell[R]:
        other = _unwrap(other)
        return _wrap(self._series.__rmul__(other))

    def __pow__(self, other: float | ExperimentalPolarsCell[P]) -> ExperimentalPolarsCell[R]:
        other = _unwrap(other)
        return _wrap(self._series.__pow__(other))

    def __rpow__(self, other: float | ExperimentalPolarsCell[P]) -> ExperimentalPolarsCell[R]:
        other = _unwrap(other)
        return _wrap(self._series.__rpow__(other))

    def __sub__(self, other: Any) -> ExperimentalPolarsCell[R]:
        other = _unwrap(other)
        return _wrap(self._series.__sub__(other))

    def __rsub__(self, other: Any) -> ExperimentalPolarsCell[R]:
        other = _unwrap(other)
        return _wrap(self._series.__rsub__(other))

    def __truediv__(self, other: Any) -> ExperimentalPolarsCell[R]:
        other = _unwrap(other)
        return _wrap(self._series.__truediv__(other))

    def __rtruediv__(self, other: Any) -> ExperimentalPolarsCell[R]:
        other = _unwrap(other)
        return _wrap(self._series.__rtruediv__(other))

    # String representation ----------------------------------------------------

    def __repr__(self) -> str:
        return self._series.__repr__()

    def __str__(self) -> str:
        return self._series.__str__()

    # Other --------------------------------------------------------------------

    def __hash__(self) -> int:
        raise NotImplementedError

    def __sizeof__(self) -> int:
        raise NotImplementedError


def _normalize_boolean_operation_operands(
    left_operand: ExperimentalPolarsColumn,
    right_operand: bool | ExperimentalPolarsCell[bool],
) -> Series | None:
    """
    Normalize the operands of a boolean operation (not, and, or, xor).

    If one of the operands is invalid, None is returned. Otherwise, the normalized right operand is returned.
    """
    import polars as pl

    if left_operand._series.dtype != pl.Boolean:
        return None
    elif isinstance(right_operand, bool):
        return pl.Series("", [right_operand])
    elif not isinstance(right_operand, ExperimentalPolarsColumn) or right_operand._series.dtype != pl.Boolean:
        return None
    else:
        return right_operand._series


def _wrap(other: Series) -> Any:
    return ExperimentalPolarsColumn._from_polars_series(other)


def _unwrap(other: Any) -> Any:
    if isinstance(other, ExperimentalPolarsColumn):
        return other._series
    return other
