from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import polars as pl
from IPython.core.display_functions import DisplayHandle, display

from safeds.data.tabular.exceptions import UnknownColumnNameError
from safeds.data.tabular.typing import ColumnType, Schema

if TYPE_CHECKING:
    from collections.abc import Iterator


class Row:
    """
    A row is a collection of values, where each value is associated with a column name.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Row:
        """
        Create a row from a dictionary that maps column names to column values.

        Parameters
        ----------
        data : dict[str, Any]
            The data.

        Returns
        -------
        row : Row
            The generated row.
        """
        return Row(pl.DataFrame(data))

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, data: pl.DataFrame, schema: Schema | None = None):
        self._data: pl.DataFrame = data

        self._schema: Schema
        if schema is not None:
            self._schema = schema
        else:
            # noinspection PyProtectedMember
            self._schema = Schema._from_polars_dataframe(self._data)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Row):
            return NotImplemented
        if self is other:
            return True
        return self._schema == other._schema and self._data.frame_equal(other._data)

    def __getitem__(self, column_name: str) -> Any:
        return self.get_value(column_name)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.get_column_names())

    def __len__(self) -> int:
        return self._data.shape[1]

    def __repr__(self) -> str:
        return f"Row({str(self)})"

    def __str__(self) -> str:
        match len(self):
            case 0:
                return "{}"
            case 1:
                return str(self.to_dict())
            case _:
                lines = (f"    {name!r}: {value!r}" for name, value in self.to_dict().items())
                joined = ",\n".join(lines)
                return f"{{\n{joined}\n}}"

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def schema(self) -> Schema:
        """
        Return the schema of the row.

        Returns
        -------
        schema : Schema
            The schema.
        """
        return self._schema

    # ------------------------------------------------------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------------------------------------------------------

    def get_value(self, column_name: str) -> Any:
        """
        Return the value of a specified column.

        Parameters
        ----------
        column_name : str
            The column name.

        Returns
        -------
        value :
            The value of the column.

        Raises
        ------
        UnknownColumnNameError
            If the row does not contain the specified column.
        """
        if not self.has_column(column_name):
            raise UnknownColumnNameError([column_name])

        return self._data[0, column_name]

    def has_column(self, column_name: str) -> bool:
        """
        Return whether the row contains a given column.

        Alias for self.schema.hasColumn(column_name: str) -> bool.

        Parameters
        ----------
        column_name : str
            The name of the column.

        Returns
        -------
        contains : bool
            True, if row contains the column.
        """
        return self._schema.has_column(column_name)

    def get_column_names(self) -> list[str]:
        """
        Return a list of all column names saved in this schema.

        Alias for self.schema.get_column_names() -> list[str].

        Returns
        -------
        column_names : list[str]
            The column names.
        """
        return self._schema.get_column_names()

    def get_type_of_column(self, column_name: str) -> ColumnType:
        """
        Return the type of the specified column.

        Parameters
        ----------
        column_name : str
            The name of the column.

        Returns
        -------
        type : ColumnType
            The type of the column.

        Raises
        ------
        ColumnNameError
            If the specified target column name does not exist.
        """
        return self._schema.get_type_of_column(column_name)

    # ------------------------------------------------------------------------------------------------------------------
    # Information
    # ------------------------------------------------------------------------------------------------------------------

    def count(self) -> int:
        """
        Return the number of columns in this row.

        Returns
        -------
        count : int
            The number of columns.
        """
        return self._data.shape[1]

    # ------------------------------------------------------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Return a dictionary that maps column names to column values.

        Returns
        -------
        data : dict[str, Any]
            Dictionary representation of the row.
        """
        return {column_name: self.get_value(column_name) for column_name in self.get_column_names()}

    # ------------------------------------------------------------------------------------------------------------------
    # IPython integration
    # ------------------------------------------------------------------------------------------------------------------

    def _ipython_display_(self) -> DisplayHandle:
        """
        Return a display object for the column to be used in Jupyter Notebooks.

        Returns
        -------
        output : DisplayHandle
            Output object.
        """
        tmp = self._data.to_pandas()

        with pd.option_context("display.max_rows", tmp.shape[0], "display.max_columns", tmp.shape[1]):
            return display(tmp)
