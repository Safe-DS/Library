from __future__ import annotations

from typing import TYPE_CHECKING

from ._experimental_polars_row import ExperimentalPolarsRow

if TYPE_CHECKING:
    from safeds.data.tabular.typing import ColumnType, Schema

    from ._experimental_polars_column import ExperimentalPolarsColumn
    from ._experimental_polars_table import ExperimentalPolarsTable


class _VectorizedRow(ExperimentalPolarsRow):
    """
    A row is a one-dimensional collection of named, heterogeneous values.

    This implementation treats an entire table as a row, where each column is a "cell" in the row. This greatly speeds
    up operations on the row.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, table: ExperimentalPolarsTable):
        self._table: ExperimentalPolarsTable = table

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _VectorizedRow):
            return NotImplemented
        if self is other:
            return True
        return self._table == other._table

    def __hash__(self) -> int:
        return self._table.__hash__()

    def __sizeof__(self) -> int:
        return self._table.__sizeof__()

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def column_names(self) -> list[str]:
        """The names of the columns in the row."""
        return self._table.column_names

    @property
    def number_of_columns(self) -> int:
        """The number of columns in the row."""
        return self._table.number_of_columns

    @property
    def schema(self) -> Schema:  # TODO: rethink return type
        """The schema of the row."""
        return self._table.schema

    # ------------------------------------------------------------------------------------------------------------------
    # Column operations
    # ------------------------------------------------------------------------------------------------------------------

    def get_value(self, name: str) -> ExperimentalPolarsColumn:
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
        return self._table.get_column(name)

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
        return self._table.get_column_type(name)

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
        return self._table.has_column(name)
