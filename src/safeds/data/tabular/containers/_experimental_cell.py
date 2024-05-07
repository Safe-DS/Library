from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from safeds.data.tabular.typing import ColumnType

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
    def __invert__(self) -> ExperimentalPolarsCell[bool]: ...

    @abstractmethod
    def __and__(self, other: bool | ExperimentalPolarsCell[bool]) -> ExperimentalPolarsCell[bool]: ...

    @abstractmethod
    def __rand__(self, other: bool | ExperimentalPolarsCell[bool]) -> ExperimentalPolarsCell[bool]: ...

    @abstractmethod
    def __or__(self, other: bool | ExperimentalPolarsCell[bool]) -> ExperimentalPolarsCell[bool]: ...

    @abstractmethod
    def __ror__(self, other: bool | ExperimentalPolarsCell[bool]) -> ExperimentalPolarsCell[bool]: ...

    @abstractmethod
    def __xor__(self, other: bool | ExperimentalPolarsCell[bool]) -> ExperimentalPolarsCell[bool]: ...

    @abstractmethod
    def __rxor__(self, other: bool | ExperimentalPolarsCell[bool]) -> ExperimentalPolarsCell[bool]: ...

    # Comparison ---------------------------------------------------------------

    @abstractmethod
    def __eq__(self, other: object) -> ExperimentalPolarsCell[bool]:  # type: ignore[override]
        ...

    @abstractmethod
    def __ge__(self, other: Any) -> ExperimentalPolarsCell[bool]: ...

    @abstractmethod
    def __gt__(self, other: Any) -> ExperimentalPolarsCell[bool]: ...

    @abstractmethod
    def __le__(self, other: Any) -> ExperimentalPolarsCell[bool]: ...

    @abstractmethod
    def __lt__(self, other: Any) -> ExperimentalPolarsCell[bool]: ...

    @abstractmethod
    def __ne__(self, other: object) -> ExperimentalPolarsCell[bool]:  # type: ignore[override]
        ...

    # Numeric operators --------------------------------------------------------

    @abstractmethod
    def __abs__(self) -> ExperimentalPolarsCell[R]: ...

    @abstractmethod
    def __neg__(self) -> ExperimentalPolarsCell[R]: ...

    @abstractmethod
    def __pos__(self) -> ExperimentalPolarsCell[R]: ...

    @abstractmethod
    def __add__(self, other: Any) -> ExperimentalPolarsCell[R]: ...

    @abstractmethod
    def __radd__(self, other: Any) -> ExperimentalPolarsCell[R]: ...

    @abstractmethod
    def __floordiv__(self, other: Any) -> ExperimentalPolarsCell[R]: ...

    @abstractmethod
    def __rfloordiv__(self, other: Any) -> ExperimentalPolarsCell[R]: ...

    @abstractmethod
    def __mod__(self, other: Any) -> ExperimentalPolarsCell[R]: ...

    @abstractmethod
    def __rmod__(self, other: Any) -> ExperimentalPolarsCell[R]: ...

    @abstractmethod
    def __mul__(self, other: Any) -> ExperimentalPolarsCell[R]: ...

    @abstractmethod
    def __rmul__(self, other: Any) -> ExperimentalPolarsCell[R]: ...

    @abstractmethod
    def __pow__(self, other: float | ExperimentalPolarsCell[P]) -> ExperimentalPolarsCell[R]: ...

    @abstractmethod
    def __rpow__(self, other: float | ExperimentalPolarsCell[P]) -> ExperimentalPolarsCell[R]: ...

    @abstractmethod
    def __sub__(self, other: Any) -> ExperimentalPolarsCell[R]: ...

    @abstractmethod
    def __rsub__(self, other: Any) -> ExperimentalPolarsCell[R]: ...

    @abstractmethod
    def __truediv__(self, other: Any) -> ExperimentalPolarsCell[R]: ...

    @abstractmethod
    def __rtruediv__(self, other: Any) -> ExperimentalPolarsCell[R]: ...

    # Other --------------------------------------------------------------------

    @abstractmethod
    def __hash__(self) -> int: ...

    @abstractmethod
    def __sizeof__(self) -> int: ...

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    @abstractmethod
    def type(self) -> ColumnType:  # TODO: rethink return type
        """The type of the column."""

    # ------------------------------------------------------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def _equals(self, other: object) -> bool:
        """
        Check if this cell is equal to another object.

        This method is needed because the `__eq__` method is used for element-wise comparisons.
        """
