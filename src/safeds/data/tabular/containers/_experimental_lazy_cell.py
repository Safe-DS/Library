from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from safeds._utils import _structural_hash

from ._experimental_cell import ExperimentalCell

if TYPE_CHECKING:
    import polars as pl

T = TypeVar("T")
P = TypeVar("P")
R = TypeVar("R")


class _LazyCell(ExperimentalCell[T]):
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

    def __invert__(self) -> ExperimentalCell[bool]:
        return _wrap(self._expression.__invert__())

    def __and__(self, other: bool | ExperimentalCell[bool]) -> ExperimentalCell[bool]:
        return _wrap(self._expression.__and__(other))

    def __rand__(self, other: bool | ExperimentalCell[bool]) -> ExperimentalCell[bool]:
        return _wrap(self._expression.__rand__(other))

    def __or__(self, other: bool | ExperimentalCell[bool]) -> ExperimentalCell[bool]:
        return _wrap(self._expression.__or__(other))

    def __ror__(self, other: bool | ExperimentalCell[bool]) -> ExperimentalCell[bool]:
        return _wrap(self._expression.__ror__(other))

    def __xor__(self, other: bool | ExperimentalCell[bool]) -> ExperimentalCell[bool]:
        return _wrap(self._expression.__xor__(other))

    def __rxor__(self, other: bool | ExperimentalCell[bool]) -> ExperimentalCell[bool]:
        return _wrap(self._expression.__rxor__(other))

    # Comparison ---------------------------------------------------------------

    def __eq__(self, other: object) -> ExperimentalCell[bool]:  # type: ignore[override]
        other = _unwrap(other)
        return _wrap(self._expression.__eq__(other))

    def __ge__(self, other: Any) -> ExperimentalCell[bool]:
        other = _unwrap(other)
        return _wrap(self._expression.__ge__(other))

    def __gt__(self, other: Any) -> ExperimentalCell[bool]:
        other = _unwrap(other)
        return _wrap(self._expression.__gt__(other))

    def __le__(self, other: Any) -> ExperimentalCell[bool]:
        other = _unwrap(other)
        return _wrap(self._expression.__le__(other))

    def __lt__(self, other: Any) -> ExperimentalCell[bool]:
        other = _unwrap(other)
        return _wrap(self._expression.__lt__(other))

    def __ne__(self, other: object) -> ExperimentalCell[bool]:  # type: ignore[override]
        other = _unwrap(other)
        return _wrap(self._expression.__ne__(other))

    # Numeric operators --------------------------------------------------------

    def __abs__(self) -> ExperimentalCell[R]:
        return _wrap(self._expression.__abs__())

    def __ceil__(self) -> ExperimentalCell[R]:
        return _wrap(self._expression.ceil())

    def __floor__(self) -> ExperimentalCell[R]:
        return _wrap(self._expression.floor())

    def __neg__(self) -> ExperimentalCell[R]:
        return _wrap(self._expression.__neg__())

    def __pos__(self) -> ExperimentalCell[R]:
        return _wrap(self._expression.__pos__())

    def __add__(self, other: Any) -> ExperimentalCell[R]:
        other = _unwrap(other)
        return _wrap(self._expression.__add__(other))

    def __radd__(self, other: Any) -> ExperimentalCell[R]:
        other = _unwrap(other)
        return _wrap(self._expression.__radd__(other))

    def __floordiv__(self, other: Any) -> ExperimentalCell[R]:
        other = _unwrap(other)
        return _wrap(self._expression.__floordiv__(other))

    def __rfloordiv__(self, other: Any) -> ExperimentalCell[R]:
        other = _unwrap(other)
        return _wrap(self._expression.__rfloordiv__(other))

    def __mod__(self, other: Any) -> ExperimentalCell[R]:
        other = _unwrap(other)
        return _wrap(self._expression.__mod__(other))

    def __rmod__(self, other: Any) -> ExperimentalCell[R]:
        other = _unwrap(other)
        return _wrap(self._expression.__rmod__(other))

    def __mul__(self, other: Any) -> ExperimentalCell[R]:
        other = _unwrap(other)
        return _wrap(self._expression.__mul__(other))

    def __rmul__(self, other: Any) -> ExperimentalCell[R]:
        other = _unwrap(other)
        return _wrap(self._expression.__rmul__(other))

    def __pow__(self, other: float | ExperimentalCell[P]) -> ExperimentalCell[R]:
        other = _unwrap(other)
        return _wrap(self._expression.__pow__(other))

    def __rpow__(self, other: float | ExperimentalCell[P]) -> ExperimentalCell[R]:
        other = _unwrap(other)
        return _wrap(self._expression.__rpow__(other))

    def __sub__(self, other: Any) -> ExperimentalCell[R]:
        other = _unwrap(other)
        return _wrap(self._expression.__sub__(other))

    def __rsub__(self, other: Any) -> ExperimentalCell[R]:
        other = _unwrap(other)
        return _wrap(self._expression.__rsub__(other))

    def __truediv__(self, other: Any) -> ExperimentalCell[R]:
        other = _unwrap(other)
        return _wrap(self._expression.__truediv__(other))

    def __rtruediv__(self, other: Any) -> ExperimentalCell[R]:
        other = _unwrap(other)
        return _wrap(self._expression.__rtruediv__(other))

    # String representation ----------------------------------------------------

    def __repr__(self) -> str:
        return self._expression.__repr__()

    def __str__(self) -> str:
        return self._expression.__str__()

    # Other --------------------------------------------------------------------

    def __hash__(self) -> int:
        return _structural_hash(self._expression.meta.serialize())

    def __sizeof__(self) -> int:
        return self._expression.__sizeof__()

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
        return self._expression.meta.eq(other._expression.meta)


def _wrap(other: pl.Expr) -> Any:
    return _LazyCell(other)


def _unwrap(other: Any) -> Any:
    if isinstance(other, _LazyCell):
        return other._expression
    return other
