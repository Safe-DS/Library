from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

T = TypeVar("T")
P = TypeVar("P")
R = TypeVar("R")


class ExperimentalPolarsCell(ABC, Generic[T]):
    """A cell is a single value in a table."""

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    # "Boolean" operators (actually bitwise) -----------------------------------

    @abstractmethod
    def __invert__(self) -> ExperimentalPolarsCell[bool]:
        ...

    @abstractmethod
    def __and__(self, other: bool | ExperimentalPolarsCell[bool]) -> ExperimentalPolarsCell[bool]:
        ...

    @abstractmethod
    def __rand__(self, other: bool | ExperimentalPolarsCell[bool]) -> ExperimentalPolarsCell[bool]:
        ...

    @abstractmethod
    def __or__(self, other: bool | ExperimentalPolarsCell[bool]) -> ExperimentalPolarsCell[bool]:
        ...

    @abstractmethod
    def __ror__(self, other: bool | ExperimentalPolarsCell[bool]) -> ExperimentalPolarsCell[bool]:
        ...

    @abstractmethod
    def __xor__(self, other: bool | ExperimentalPolarsCell[bool]) -> ExperimentalPolarsCell[bool]:
        ...

    @abstractmethod
    def __rxor__(self, other: bool | ExperimentalPolarsCell[bool]) -> ExperimentalPolarsCell[bool]:
        ...

    # Comparison ---------------------------------------------------------------

    @abstractmethod
    def __eq__(self, other: object) -> ExperimentalPolarsCell[bool]:
        ...

    @abstractmethod
    def __ge__(self, other: Any) -> ExperimentalPolarsCell[bool]:
        ...

    @abstractmethod
    def __gt__(self, other: Any) -> ExperimentalPolarsCell[bool]:
        ...

    @abstractmethod
    def __le__(self, other: Any) -> ExperimentalPolarsCell[bool]:
        ...

    @abstractmethod
    def __lt__(self, other: Any) -> ExperimentalPolarsCell[bool]:
        ...

    # Numeric operators --------------------------------------------------------

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
    def __add__(self, other: Any) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __radd__(self, other: Any) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __floordiv__(self, other: Any) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __rfloordiv__(self, other: Any) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __mod__(self, other: Any) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __rmod__(self, other: Any) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __mul__(self, other: Any) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __rmul__(self, other: Any) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __pow__(self, other: float | ExperimentalPolarsCell[P]) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __rpow__(self, other: float | ExperimentalPolarsCell[P]) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __sub__(self, other: Any) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __rsub__(self, other: Any) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __truediv__(self, other: Any) -> ExperimentalPolarsCell[R]:
        ...

    @abstractmethod
    def __rtruediv__(self, other: Any) -> ExperimentalPolarsCell[R]:
        ...

    # Other --------------------------------------------------------------------

    @abstractmethod
    def __hash__(self) -> int:
        ...

    @abstractmethod
    def __sizeof__(self) -> int:
        ...
