from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from safeds.data.tabular.containers import Cell


class MathOperations(ABC):
    """
    Namespace for mathematical operations.

    This class cannot be instantiated directly. It can only be accessed using the `math` attribute of a cell.
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
