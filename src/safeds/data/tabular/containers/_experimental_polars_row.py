from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from collections.abc import Mapping, Iterator

from safeds.data.tabular.containers import ExperimentalPolarsCell
from safeds.data.tabular.typing import Schema, ColumnType


class ExperimentalPolarsRow(ABC, Mapping[str, Any]):
    """A row is a one-dimensional collection of named, heterogeneous values."""

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __contains__(self, name: Any) -> bool:
        return self.has_column(name)

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        ...

    def __getitem__(self, name: str) -> ExperimentalPolarsCell:
        return self.get_value(name)

    @abstractmethod
    def __hash__(self) -> int:
        ...

    def __iter__(self) -> Iterator[Any]:
        return iter(self.column_names)

    def __len__(self) -> int:
        return self.number_of_columns

    @abstractmethod
    def __sizeof__(self) -> int:
        ...

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    @abstractmethod
    def column_names(self) -> list[str]:
        """The names of the columns in the row."""

    @property
    @abstractmethod
    def number_of_columns(self) -> int:
        """The number of columns in the row."""

    @property
    @abstractmethod
    def schema(self) -> Schema:  # TODO: rethink return type
        """The schema of the row."""

    # ------------------------------------------------------------------------------------------------------------------
    # Column operations
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def get_value(self, name: str) -> ExperimentalPolarsCell:
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
        """

    @abstractmethod
    def get_column_type(self, name: str) -> ColumnType:  # TODO: rethink return type
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
