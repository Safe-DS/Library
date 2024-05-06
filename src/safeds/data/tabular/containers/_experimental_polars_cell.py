from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class ExperimentalPolarsCell(ABC, Generic[T]):
    """A cell is a single value in a table."""

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    # Comparison ---------------------------------------------------------------

    @abstractmethod
    def __eq__(self, other: object) -> ExperimentalPolarsCell[bool]:
        ...

    @abstractmethod
    def __ge__(self, other) -> ExperimentalPolarsCell[bool]:
        ...

    @abstractmethod
    def __gt__(self, other) -> ExperimentalPolarsCell[bool]:
        ...

    @abstractmethod
    def __le__(self, other) -> ExperimentalPolarsCell[bool]:
        ...

    @abstractmethod
    def __lt__(self, other) -> ExperimentalPolarsCell[bool]:
        ...

    # Numeric operators (left operand) -----------------------------------------

    @abstractmethod
    def __add__(self, other) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __and__(self, other) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __floordiv__(self, other) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __matmul__(self, other) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __mod__(self, other) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __mul__(self, other) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __or__(self, other) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __pow__(self, other) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __sub__(self, other) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __truediv__(self, other) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __xor__(self, other) -> ExperimentalPolarsCell[R]:
        ...

    # Numeric operators (right operand) ----------------------------------------

    @abstractmethod
    def __radd__(self, other) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __rand__(self, other) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __rfloordiv__(self, other) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __rmatmul__(self, other) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __rmod__(self, other) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __rmul__(self, other) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __ror__(self, other) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __rpow__(self, other) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __rsub__(self, other) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __rtruediv__(self, other) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __rxor__(self, other) -> ExperimentalPolarsCell[R]:
        ...

    # Unary operators ----------------------------------------------------------

    @abstractmethod
    def __abs__(self) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __neg__(self) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __pos__(self) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __invert__(self) -> ExperimentalPolarsCell[R]:
        ...

    # Other --------------------------------------------------------------------

    @abstractmethod
    def __hash__(self) -> int:
        ...

    @abstractmethod
    def __sizeof__(self) -> int:
        ...
