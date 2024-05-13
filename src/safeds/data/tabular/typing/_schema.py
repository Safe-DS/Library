from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._data_type import DataType


class Schema(ABC):
    """The schema of a row or table."""

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
    def column_names(self) -> list[str]:
        """
        Return a list of all column names contained in this schema.

        Returns
        -------
        column_names:
            The column names.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"A": [1, 2, 3], "B": ["a", "b", "c"]})
        >>> table.schema.column_names
        ['A', 'B']
        """

    # ------------------------------------------------------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def get_column_type(self, name: str) -> DataType:
        """
        Return the type of the given column.

        Parameters
        ----------
        name:
            The name of the column.

        Returns
        -------
        type:
            The type of the column.

        Raises
        ------
        ColumnNotFoundError
            If the specified column name does not exist.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"A": [1, 2, 3], "B": ["a", "b", "c"]})
        >>> type_ = table.schema.get_column_type("A")
        """

    @abstractmethod
    def has_column(self, name: str) -> bool:
        """
        Return whether the schema contains a given column.

        Parameters
        ----------
        name:
            The name of the column.

        Returns
        -------
        contains:
            True if the schema contains the column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"A": [1, 2, 3], "B": ["a", "b", "c"]})
        >>> table.schema.has_column("A")
        True

        >>> table.schema.has_column("C")
        False
        """

    # ------------------------------------------------------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def to_dict(self) -> dict[str, DataType]:
        """
        Return a dictionary that maps column names to column types.

        Returns
        -------
        data:
            Dictionary representation of the schema.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"A": [1, 2, 3], "B": ["a", "b", "c"]})
        >>> dict_ = table.schema.to_dict()
        """

    # ------------------------------------------------------------------------------------------------------------------
    # IPython integration
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def _repr_markdown_(self) -> str:
        """
        Return a Markdown representation of the schema for IPython.

        Returns
        -------
        markdown:
            The generated Markdown.
        """
