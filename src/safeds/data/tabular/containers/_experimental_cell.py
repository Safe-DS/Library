from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

T = TypeVar("T")
P = TypeVar("P")
R = TypeVar("R")


class ExperimentalCell(ABC, Generic[T]):
    """
    A single value in a table.

    This class cannot be instantiated directly. It is only used for arguments of callbacks.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    # "Boolean" operators (actually bitwise) -----------------------------------

    @abstractmethod
    def __invert__(self) -> ExperimentalCell[bool]: ...

    @abstractmethod
    def __and__(self, other: bool | ExperimentalCell[bool]) -> ExperimentalCell[bool]: ...

    @abstractmethod
    def __rand__(self, other: bool | ExperimentalCell[bool]) -> ExperimentalCell[bool]: ...

    @abstractmethod
    def __or__(self, other: bool | ExperimentalCell[bool]) -> ExperimentalCell[bool]: ...

    @abstractmethod
    def __ror__(self, other: bool | ExperimentalCell[bool]) -> ExperimentalCell[bool]: ...

    @abstractmethod
    def __xor__(self, other: bool | ExperimentalCell[bool]) -> ExperimentalCell[bool]: ...

    @abstractmethod
    def __rxor__(self, other: bool | ExperimentalCell[bool]) -> ExperimentalCell[bool]: ...

    # Comparison ---------------------------------------------------------------

    @abstractmethod
    def __eq__(self, other: object) -> ExperimentalCell[bool]:  # type: ignore[override]
        ...

    @abstractmethod
    def __ge__(self, other: Any) -> ExperimentalCell[bool]: ...

    @abstractmethod
    def __gt__(self, other: Any) -> ExperimentalCell[bool]: ...

    @abstractmethod
    def __le__(self, other: Any) -> ExperimentalCell[bool]: ...

    @abstractmethod
    def __lt__(self, other: Any) -> ExperimentalCell[bool]: ...

    @abstractmethod
    def __ne__(self, other: object) -> ExperimentalCell[bool]:  # type: ignore[override]
        ...

    # Numeric operators --------------------------------------------------------

    @abstractmethod
    def __abs__(self) -> ExperimentalCell[R]: ...

    @abstractmethod
    def __neg__(self) -> ExperimentalCell[R]: ...

    @abstractmethod
    def __pos__(self) -> ExperimentalCell[R]: ...

    @abstractmethod
    def __add__(self, other: Any) -> ExperimentalCell[R]: ...

    @abstractmethod
    def __radd__(self, other: Any) -> ExperimentalCell[R]: ...

    @abstractmethod
    def __floordiv__(self, other: Any) -> ExperimentalCell[R]: ...

    @abstractmethod
    def __rfloordiv__(self, other: Any) -> ExperimentalCell[R]: ...

    @abstractmethod
    def __mod__(self, other: Any) -> ExperimentalCell[R]: ...

    @abstractmethod
    def __rmod__(self, other: Any) -> ExperimentalCell[R]: ...

    @abstractmethod
    def __mul__(self, other: Any) -> ExperimentalCell[R]: ...

    @abstractmethod
    def __rmul__(self, other: Any) -> ExperimentalCell[R]: ...

    @abstractmethod
    def __pow__(self, other: float | ExperimentalCell[P]) -> ExperimentalCell[R]: ...

    @abstractmethod
    def __rpow__(self, other: float | ExperimentalCell[P]) -> ExperimentalCell[R]: ...

    @abstractmethod
    def __sub__(self, other: Any) -> ExperimentalCell[R]: ...

    @abstractmethod
    def __rsub__(self, other: Any) -> ExperimentalCell[R]: ...

    @abstractmethod
    def __truediv__(self, other: Any) -> ExperimentalCell[R]: ...

    @abstractmethod
    def __rtruediv__(self, other: Any) -> ExperimentalCell[R]: ...

    # Other --------------------------------------------------------------------

    @abstractmethod
    def __hash__(self) -> int: ...

    @abstractmethod
    def __sizeof__(self) -> int: ...

    # ------------------------------------------------------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def _equals(self, other: object) -> bool:
        """
        Check if this cell is equal to another object.

        This method is needed because the `__eq__` method is used for element-wise comparisons.
        """
