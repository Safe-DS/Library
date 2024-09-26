from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    import polars as pl

    from ._string_cell import StringCell
    from ._temporal_cell import TemporalCell

T_co = TypeVar("T_co", covariant=True)
P_contra = TypeVar("P_contra", contravariant=True)
R_co = TypeVar("R_co", covariant=True)


class Cell(ABC, Generic[T_co]):
    """
    A single value in a table.

    This class cannot be instantiated directly. It is only used for arguments of callbacks.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Static methods
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def first_not_none(cells: list[Cell]) -> Cell:
        """
        Return the first cell from the given list that is not None.

        Parameters
        ----------
        cells:
            The list of cells to be searched.

        Returns
        -------
        cell:
            Returns the contents of the first cell that is not None.
            If all cells in the list are None or the list is empty returns None.
        """
        import polars as pl

        from ._lazy_cell import _LazyCell  # circular import

        return _LazyCell(pl.coalesce([cell._polars_expression for cell in cells]))

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    # "Boolean" operators (actually bitwise) -----------------------------------

    @abstractmethod
    def __invert__(self) -> Cell[bool]: ...

    @abstractmethod
    def __and__(self, other: bool | Cell[bool]) -> Cell[bool]: ...

    @abstractmethod
    def __rand__(self, other: bool | Cell[bool]) -> Cell[bool]: ...

    @abstractmethod
    def __or__(self, other: bool | Cell[bool]) -> Cell[bool]: ...

    @abstractmethod
    def __ror__(self, other: bool | Cell[bool]) -> Cell[bool]: ...

    @abstractmethod
    def __xor__(self, other: bool | Cell[bool]) -> Cell[bool]: ...

    @abstractmethod
    def __rxor__(self, other: bool | Cell[bool]) -> Cell[bool]: ...

    # Comparison ---------------------------------------------------------------

    @abstractmethod
    def __eq__(self, other: object) -> Cell[bool]:  # type: ignore[override]
        ...

    @abstractmethod
    def __ge__(self, other: Any) -> Cell[bool]: ...

    @abstractmethod
    def __gt__(self, other: Any) -> Cell[bool]: ...

    @abstractmethod
    def __le__(self, other: Any) -> Cell[bool]: ...

    @abstractmethod
    def __lt__(self, other: Any) -> Cell[bool]: ...

    @abstractmethod
    def __ne__(self, other: object) -> Cell[bool]:  # type: ignore[override]
        ...

    # Numeric operators --------------------------------------------------------

    @abstractmethod
    def __abs__(self) -> Cell[R_co]: ...

    @abstractmethod
    def __ceil__(self) -> Cell[R_co]: ...

    @abstractmethod
    def __floor__(self) -> Cell[R_co]: ...

    @abstractmethod
    def __neg__(self) -> Cell[R_co]: ...

    @abstractmethod
    def __pos__(self) -> Cell[R_co]: ...

    @abstractmethod
    def __add__(self, other: Any) -> Cell[R_co]: ...

    @abstractmethod
    def __radd__(self, other: Any) -> Cell[R_co]: ...

    @abstractmethod
    def __floordiv__(self, other: Any) -> Cell[R_co]: ...

    @abstractmethod
    def __rfloordiv__(self, other: Any) -> Cell[R_co]: ...

    @abstractmethod
    def __mod__(self, other: Any) -> Cell[R_co]: ...

    @abstractmethod
    def __rmod__(self, other: Any) -> Cell[R_co]: ...

    @abstractmethod
    def __mul__(self, other: Any) -> Cell[R_co]: ...

    @abstractmethod
    def __rmul__(self, other: Any) -> Cell[R_co]: ...

    @abstractmethod
    def __pow__(self, other: float | Cell[P_contra]) -> Cell[R_co]: ...

    @abstractmethod
    def __rpow__(self, other: float | Cell[P_contra]) -> Cell[R_co]: ...

    @abstractmethod
    def __sub__(self, other: Any) -> Cell[R_co]: ...

    @abstractmethod
    def __rsub__(self, other: Any) -> Cell[R_co]: ...

    @abstractmethod
    def __truediv__(self, other: Any) -> Cell[R_co]: ...

    @abstractmethod
    def __rtruediv__(self, other: Any) -> Cell[R_co]: ...

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
    def str(self) -> StringCell:
        """Namespace for operations on strings."""

    @property
    @abstractmethod
    def dt(self) -> TemporalCell:
        """Namespace for operations on date time values."""

    # ------------------------------------------------------------------------------------------------------------------
    # Boolean operations
    # ------------------------------------------------------------------------------------------------------------------

    def not_(self) -> Cell[bool]:
        """
        Negate a boolean. This is equivalent to the `~` operator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", [True, False])
        >>> column.transform(lambda cell: cell.not_())
        +---------+
        | example |
        | ---     |
        | bool    |
        +=========+
        | false   |
        | true    |
        +---------+

        >>> column.transform(lambda cell: ~cell)
        +---------+
        | example |
        | ---     |
        | bool    |
        +=========+
        | false   |
        | true    |
        +---------+
        """
        return self.__invert__()

    def and_(self, other: bool | Cell[bool]) -> Cell[bool]:
        """
        Perform a boolean AND operation. This is equivalent to the `&` operator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", [True, False])
        >>> column.transform(lambda cell: cell.and_(False))
        +---------+
        | example |
        | ---     |
        | bool    |
        +=========+
        | false   |
        | false   |
        +---------+

        >>> column.transform(lambda cell: cell & False)
        +---------+
        | example |
        | ---     |
        | bool    |
        +=========+
        | false   |
        | false   |
        +---------+
        """
        return self.__and__(other)

    def or_(self, other: bool | Cell[bool]) -> Cell[bool]:
        """
        Perform a boolean OR operation. This is equivalent to the `|` operator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", [True, False])
        >>> column.transform(lambda cell: cell.or_(True))
        +---------+
        | example |
        | ---     |
        | bool    |
        +=========+
        | true    |
        | true    |
        +---------+

        >>> column.transform(lambda cell: cell | True)
        +---------+
        | example |
        | ---     |
        | bool    |
        +=========+
        | true    |
        | true    |
        +---------+
        """
        return self.__or__(other)

    def xor(self, other: bool | Cell[bool]) -> Cell[bool]:
        """
        Perform a boolean XOR operation. This is equivalent to the `^` operator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", [True, False])
        >>> column.transform(lambda cell: cell.xor(True))
        +---------+
        | example |
        | ---     |
        | bool    |
        +=========+
        | false   |
        | true    |
        +---------+

        >>> column.transform(lambda cell: cell ^ True)
        +---------+
        | example |
        | ---     |
        | bool    |
        +=========+
        | false   |
        | true    |
        +---------+
        """
        return self.__xor__(other)

    # ------------------------------------------------------------------------------------------------------------------
    # Numeric operations
    # ------------------------------------------------------------------------------------------------------------------

    def abs(self) -> Cell[R_co]:
        """
        Get the absolute value.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", [1, -2])
        >>> column.transform(lambda cell: cell.abs())
        +---------+
        | example |
        |     --- |
        |     i64 |
        +=========+
        |       1 |
        |       2 |
        +---------+
        """
        return self.__abs__()

    def ceil(self) -> Cell[R_co]:
        """
        Round up to the nearest integer.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", [1.1, 2.9])
        >>> column.transform(lambda cell: cell.ceil())
        +---------+
        | example |
        |     --- |
        |     f64 |
        +=========+
        | 2.00000 |
        | 3.00000 |
        +---------+
        """
        return self.__ceil__()

    def floor(self) -> Cell[R_co]:
        """
        Round down to the nearest integer.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", [1.1, 2.9])
        >>> column.transform(lambda cell: cell.floor())
        +---------+
        | example |
        |     --- |
        |     f64 |
        +=========+
        | 1.00000 |
        | 2.00000 |
        +---------+
        """
        return self.__floor__()

    def neg(self) -> Cell[R_co]:
        """
        Negate the value.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", [1, -2])
        >>> column.transform(lambda cell: cell.neg())
        +---------+
        | example |
        |     --- |
        |     i64 |
        +=========+
        |      -1 |
        |       2 |
        +---------+
        """
        return self.__neg__()

    def add(self, other: Any) -> Cell[R_co]:
        """
        Add a value. This is equivalent to the `+` operator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", [1, 2])
        >>> column.transform(lambda cell: cell.add(3))
        +---------+
        | example |
        |     --- |
        |     i64 |
        +=========+
        |       4 |
        |       5 |
        +---------+

        >>> column.transform(lambda cell: cell + 3)
        +---------+
        | example |
        |     --- |
        |     i64 |
        +=========+
        |       4 |
        |       5 |
        +---------+
        """
        return self.__add__(other)

    def div(self, other: Any) -> Cell[R_co]:
        """
        Divide by a value. This is equivalent to the `/` operator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", [6, 8])
        >>> column.transform(lambda cell: cell.div(2))
        +---------+
        | example |
        |     --- |
        |     f64 |
        +=========+
        | 3.00000 |
        | 4.00000 |
        +---------+

        >>> column.transform(lambda cell: cell / 2)
        +---------+
        | example |
        |     --- |
        |     f64 |
        +=========+
        | 3.00000 |
        | 4.00000 |
        +---------+
        """
        return self.__truediv__(other)

    def mod(self, other: Any) -> Cell[R_co]:
        """
        Perform a modulo operation. This is equivalent to the `%` operator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", [5, 6])
        >>> column.transform(lambda cell: cell.mod(3))
        +---------+
        | example |
        |     --- |
        |     i64 |
        +=========+
        |       2 |
        |       0 |
        +---------+

        >>> column.transform(lambda cell: cell % 3)
        +---------+
        | example |
        |     --- |
        |     i64 |
        +=========+
        |       2 |
        |       0 |
        +---------+
        """
        return self.__mod__(other)

    def mul(self, other: Any) -> Cell[R_co]:
        """
        Multiply by a value. This is equivalent to the `*` operator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", [2, 3])
        >>> column.transform(lambda cell: cell.mul(4))
        +---------+
        | example |
        |     --- |
        |     i64 |
        +=========+
        |       8 |
        |      12 |
        +---------+

        >>> column.transform(lambda cell: cell * 4)
        +---------+
        | example |
        |     --- |
        |     i64 |
        +=========+
        |       8 |
        |      12 |
        +---------+
        """
        return self.__mul__(other)

    def pow(self, other: float | Cell[P_contra]) -> Cell[R_co]:
        """
        Raise to a power. This is equivalent to the `**` operator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", [2, 3])
        >>> column.transform(lambda cell: cell.pow(3))
        +---------+
        | example |
        |     --- |
        |     i64 |
        +=========+
        |       8 |
        |      27 |
        +---------+

        >>> column.transform(lambda cell: cell ** 3)
        +---------+
        | example |
        |     --- |
        |     i64 |
        +=========+
        |       8 |
        |      27 |
        +---------+
        """
        return self.__pow__(other)

    def sub(self, other: Any) -> Cell[R_co]:
        """
        Subtract a value. This is equivalent to the `-` operator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", [5, 6])
        >>> column.transform(lambda cell: cell.sub(3))
        +---------+
        | example |
        |     --- |
        |     i64 |
        +=========+
        |       2 |
        |       3 |
        +---------+

        >>> column.transform(lambda cell: cell - 3)
        +---------+
        | example |
        |     --- |
        |     i64 |
        +=========+
        |       2 |
        |       3 |
        +---------+
        """
        return self.__sub__(other)

    # ------------------------------------------------------------------------------------------------------------------
    # Comparison operations
    # ------------------------------------------------------------------------------------------------------------------

    def eq(self, other: Any) -> Cell[bool]:
        """
        Check if equal to a value. This is equivalent to the `==` operator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", [1, 2])
        >>> column.transform(lambda cell: cell.eq(2))
        +---------+
        | example |
        | ---     |
        | bool    |
        +=========+
        | false   |
        | true    |
        +---------+

        >>> column.transform(lambda cell: cell == 2)
        +---------+
        | example |
        | ---     |
        | bool    |
        +=========+
        | false   |
        | true    |
        +---------+
        """
        return self.__eq__(other)

    def neq(self, other: Any) -> Cell[bool]:
        """
        Check if not equal to a value. This is equivalent to the `!=` operator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", [1, 2])
        >>> column.transform(lambda cell: cell.neq(2))
        +---------+
        | example |
        | ---     |
        | bool    |
        +=========+
        | true    |
        | false   |
        +---------+

        >>> column.transform(lambda cell: cell != 2)
        +---------+
        | example |
        | ---     |
        | bool    |
        +=========+
        | true    |
        | false   |
        +---------+
        """
        return self.__ne__(other)

    def ge(self, other: Any) -> Cell[bool]:
        """
        Check if greater than or equal to a value. This is equivalent to the `>=` operator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", [1, 2])
        >>> column.transform(lambda cell: cell.ge(2))
        +---------+
        | example |
        | ---     |
        | bool    |
        +=========+
        | false   |
        | true    |
        +---------+

        >>> column.transform(lambda cell: cell >= 2)
        +---------+
        | example |
        | ---     |
        | bool    |
        +=========+
        | false   |
        | true    |
        +---------+
        """
        return self.__ge__(other)

    def gt(self, other: Any) -> Cell[bool]:
        """
        Check if greater than a value. This is equivalent to the `>` operator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", [1, 2])
        >>> column.transform(lambda cell: cell.gt(2))
        +---------+
        | example |
        | ---     |
        | bool    |
        +=========+
        | false   |
        | false   |
        +---------+

        >>> column.transform(lambda cell: cell > 2)
        +---------+
        | example |
        | ---     |
        | bool    |
        +=========+
        | false   |
        | false   |
        +---------+
        """
        return self.__gt__(other)

    def le(self, other: Any) -> Cell[bool]:
        """
        Check if less than or equal to a value. This is equivalent to the `<=` operator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", [1, 2])
        >>> column.transform(lambda cell: cell.le(2))
        +---------+
        | example |
        | ---     |
        | bool    |
        +=========+
        | true    |
        | true    |
        +---------+

        >>> column.transform(lambda cell: cell <= 2)
        +---------+
        | example |
        | ---     |
        | bool    |
        +=========+
        | true    |
        | true    |
        +---------+
        """
        return self.__le__(other)

    def lt(self, other: Any) -> Cell[bool]:
        """
        Check if less than a value. This is equivalent to the `<` operator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", [1, 2])
        >>> column.transform(lambda cell: cell.lt(2))
        +---------+
        | example |
        | ---     |
        | bool    |
        +=========+
        | true    |
        | false   |
        +---------+

        >>> column.transform(lambda cell: cell < 2)
        +---------+
        | example |
        | ---     |
        | bool    |
        +=========+
        | true    |
        | false   |
        +---------+
        """
        return self.__lt__(other)

    # ------------------------------------------------------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------------------------------------------------------

    @property
    @abstractmethod
    def _polars_expression(self) -> pl.Expr:
        """The Polars expression that corresponds to this cell."""

    @abstractmethod
    def _equals(self, other: object) -> bool:
        """
        Check if this cell is equal to another object.

        This method is needed because the `__eq__` method is used for element-wise comparisons.
        """
