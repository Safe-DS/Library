import typing
from typing import Any

import pandas as pd
from IPython.core.display_functions import DisplayHandle, display
from safeds.exceptions import UnknownColumnNameError

from ._column_type import ColumnType
from ._table_schema import TableSchema


class Row:
    def __init__(self, data: typing.Iterable, schema: TableSchema):
        self._data: pd.Series = data if isinstance(data, pd.Series) else pd.Series(data)
        self.schema: TableSchema = schema
        self._data = self._data.reset_index(drop=True)

    def __getitem__(self, column_name: str) -> Any:
        return self.get_value(column_name)

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
        """
        if not self.schema.has_column(column_name):
            raise UnknownColumnNameError([column_name])
        return self._data[self.schema._get_column_index_by_name(column_name)]

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
        return self.schema.has_column(column_name)

    def get_column_names(self) -> list[str]:
        """
        Return a list of all column names saved in this schema.
        Alias for self.schema.get_column_names() -> list[str].

        Returns
        -------
        column_names : list[str]
            The column names.
        """
        return self.schema.get_column_names()

    def get_type_of_column(self, column_name: str) -> ColumnType:
        """
        Return the type of a specified column.
        Alias for self.schema.get_type_of_column(column_name: str) -> ColumnType.

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
        return self.schema.get_type_of_column(column_name)

    def __eq__(self, other: typing.Any) -> bool:
        if not isinstance(other, Row):
            return NotImplemented
        if self is other:
            return True
        return self._data.equals(other._data) and self.schema == other.schema

    def __hash__(self) -> int:
        return hash(self._data)

    def __str__(self) -> str:
        tmp = self._data.to_frame().T
        tmp.columns = self.get_column_names()
        return tmp.__str__()

    def __repr__(self) -> str:
        tmp = self._data.to_frame().T
        tmp.columns = self.get_column_names()
        return tmp.__repr__()

    def _ipython_display_(self) -> DisplayHandle:
        """
        Return a display object for the column to be used in Jupyter Notebooks.

        Returns
        -------
        output : DisplayHandle
            Output object.
        """
        tmp = self._data.to_frame().T
        tmp.columns = self.get_column_names()

        with pd.option_context(
            "display.max_rows", tmp.shape[0], "display.max_columns", tmp.shape[1]
        ):
            return display(tmp)
