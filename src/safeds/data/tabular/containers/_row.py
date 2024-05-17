from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from safeds.data.tabular.typing import DataType, Schema

    from ._cell import Cell


class Row(ABC, Mapping[str, Any]):
    """
    A one-dimensional collection of named, heterogeneous values.

    This class cannot be instantiated directly. It is only used for arguments of callbacks.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __contains__(self, name: Any) -> bool:
        return self.has_column(name)

    @abstractmethod
    def __eq__(self, other: object) -> bool: ...

    def __getitem__(self, name: str) -> Cell:
        return self.get_value(name)

    @abstractmethod
    def __hash__(self) -> int: ...

    def __iter__(self) -> Iterator[Any]:
        return iter(self.column_names)

    def __len__(self) -> int:
        return self.column_count

    @abstractmethod
    def __sizeof__(self) -> int: ...

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    @abstractmethod
    def column_names(self) -> list[str]:
        """The names of the columns in the row."""

    @property
    @abstractmethod
    def column_count(self) -> int:
        """The number of columns in the row."""

    @property
    @abstractmethod
    def schema(self) -> Schema:
        """The schema of the row."""

    # ------------------------------------------------------------------------------------------------------------------
    # Column operations
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def get_value(self, name: str) -> Cell:
        """
        Get the value of the specified column.

        Parameters
        ----------
        name:
            The name of the column.

        Returns
        -------
        value:
            The value of the column.

        Raises
        ------
        ColumnNotFoundError
            If the column name does not exist.
        """

    @abstractmethod
    def get_column_type(self, name: str) -> DataType:
        """
        Get the type of the specified column.

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
            If the column name does not exist.
        """

    @abstractmethod
    def has_column(self, name: str) -> bool:
        """
        Check if the row has a column with the specified name.

        Parameters
        ----------
        name:
            The name of the column.

        Returns
        -------
        has_column:
            Whether the row has a column with the specified name.
        """
