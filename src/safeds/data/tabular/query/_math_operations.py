from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from safeds.data.tabular.containers import Cell


class MathOperations(ABC):
    """
    Namespace for mathematical operations.

    This class cannot be instantiated directly. It can only be accessed using the `math` attribute of a cell.

    Examples
    --------
    >>> from safeds.data.tabular.containers import Column
    >>> column = Column("a", [-1, 0, 1])
    >>> column.transform(lambda cell: cell.math.abs())
    +-----+
    |   a |
    | --- |
    | i64 |
    +=====+
    |   1 |
    |   0 |
    |   1 |
    +-----+
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def __eq__(self, other: object) -> bool: ...

    @abstractmethod
    def __hash__(self) -> int: ...

    @abstractmethod
    def __repr__(self) -> str: ...

    @abstractmethod
    def __sizeof__(self) -> int: ...

    @abstractmethod
    def __str__(self) -> str: ...

    # ------------------------------------------------------------------------------------------------------------------
    # Math operations
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def abs(self) -> Cell:
        """
        Get the absolute value.

        Returns
        -------
        cell:
            The absolute value.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, -2, None])
        >>> column.transform(lambda cell: cell.math.abs())
        +------+
        |    a |
        |  --- |
        |  i64 |
        +======+
        |    1 |
        |    2 |
        | null |
        +------+
        """

    @abstractmethod
    def ceil(self) -> Cell:
        """
        Round up to the nearest integer.

        Returns
        -------
        cell:
            The rounded value.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1.1, 3.0, None])
        >>> column.transform(lambda cell: cell.math.ceil())
        +---------+
        |       a |
        |     --- |
        |     f64 |
        +=========+
        | 2.00000 |
        | 3.00000 |
        |    null |
        +---------+
        """

    @abstractmethod
    def floor(self) -> Cell:
        """
        Round down to the nearest integer.

        Returns
        -------
        cell:
            The rounded value.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1.1, 3.0, None])
        >>> column.transform(lambda cell: cell.math.floor())
        +---------+
        |       a |
        |     --- |
        |     f64 |
        +=========+
        | 1.00000 |
        | 3.00000 |
        |    null |
        +---------+
        """

    @abstractmethod
    def round_to_decimal_places(self, decimal_places: int) -> Cell:
        """
        Round to the specified number of decimal places.

        Parameters
        ----------
        decimal_places:
            The number of decimal places to round to.

        Returns
        -------
        cell:
            The rounded value.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [0.999, 1.123, 3.456, None])
        >>> column.transform(lambda cell: cell.math.round_to_decimal_places(0))
        +---------+
        |       a |
        |     --- |
        |     f64 |
        +=========+
        | 1.00000 |
        | 1.00000 |
        | 3.00000 |
        |    null |
        +---------+

        >>> column.transform(lambda cell: cell.math.round_to_decimal_places(2))
        +---------+
        |       a |
        |     --- |
        |     f64 |
        +=========+
        | 1.00000 |
        | 1.12000 |
        | 3.46000 |
        |    null |
        +---------+
        """

    @abstractmethod
    def round_to_significant_figures(self, significant_figures: int) -> Cell:
        """
        Round to the specified number of significant figures.

        Parameters
        ----------
        significant_figures:
            The number of significant figures to round to.

        Returns
        -------
        cell:
            The rounded value.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [0.999, 1.123, 3.456, None])
        >>> column.transform(lambda cell: cell.math.round_to_significant_figures(1))
        +---------+
        |       a |
        |     --- |
        |     f64 |
        +=========+
        | 1.00000 |
        | 1.00000 |
        | 3.00000 |
        |    null |
        +---------+

        >>> column.transform(lambda cell: cell.math.round_to_significant_figures(2))
        +---------+
        |       a |
        |     --- |
        |     f64 |
        +=========+
        | 1.00000 |
        | 1.10000 |
        | 3.50000 |
        |    null |
        +---------+
        """

    @abstractmethod
    def sign(self) -> Cell:
        """
        Get the sign (-1 for negative numbers, 0 for zero, and 1 for positive numbers).

        Returns
        -------
        cell:
            The sign.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [-2, 0, 2, None])
        >>> column.transform(lambda cell: cell.math.sign())
        +------+
        |    a |
        |  --- |
        |  i64 |
        +======+
        |   -1 |
        |    0 |
        |    1 |
        | null |
        +------+
        """
