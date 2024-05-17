from __future__ import annotations

from abc import ABC, abstractmethod


class DataType(ABC):
    """The type of a column or cell in a table."""

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
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    @abstractmethod
    def is_numeric(self) -> bool:
        """
        Whether the column type is numeric.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table(
        ...     {
        ...         "A": [1, 2, 3],
        ...         "B": ["a", "b", "c"]
        ...     }
        ... )
        >>> table.get_column_type("A").is_numeric
        True

        >>> table.get_column_type("B").is_numeric
        False
        """

    @property
    @abstractmethod
    def is_temporal(self) -> bool:
        """
        Whether the column type is operator.

        Examples
        --------
        >>> from datetime import datetime
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table(
        ...     {
        ...         "A": [datetime.now(), datetime.now(), datetime.now()],
        ...         "B": ["a", "b", "c"]
        ...     }
        ... )
        >>> table.get_column_type("A").is_temporal
        True

        >>> table.get_column_type("B").is_temporal
        False
        """
