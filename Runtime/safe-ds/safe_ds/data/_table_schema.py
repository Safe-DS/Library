from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from safe_ds.exceptions import UnknownColumnNameError

from ._column_type import ColumnType


@dataclass
class TableSchema:
    """Stores column names and corresponding data types for a table

    Parameters
    ----------
    schema: dict[str, ColumnType]
        Map from column names to data types
    """

    _schema: dict[str, ColumnType]

    def __init__(self, schema: dict[str, ColumnType]):
        self._schema = dict(schema)  # Defensive copy

    def has_column(self, column_name: str) -> bool:
        """
        Returns if the schema contains a given column

        Parameters
        ----------
        column_name : str
            The name of the column you want to look for

        Returns
        -------
        contains: bool
            If it contains the column
        """
        return column_name in self._schema

    def get_type_of_column(self, column_name: str) -> ColumnType:
        """
        Returns the type of the given column

        Parameters
        ----------
        column_name : str
            The name of the column you want the type of

        Returns
        -------
        type: ColumnType
            The type of the column

        Raises
        ------
        ColumnNameError
            If the specified target column name doesn't exist
        """
        if not self.has_column(column_name):
            raise UnknownColumnNameError([column_name])
        return self._schema[column_name]

    def _get_column_index_by_name(self, column_name: str) -> int:
        """
        Returns the index of the column with the given column_name

        Parameters
        ----------
        column_name: str
            The column_name you want the index for

        Returns
        -------
        The index of the column
        """

        return list(self._schema.keys()).index(column_name)

    @staticmethod
    def _from_dataframe(dataframe: pd.DataFrame) -> TableSchema:
        """
        Constructs a TableSchema from a Dataframe. This function is not supposed to be exposed to the user.

        Parameters
        ----------
        dataframe: pd.Dataframe
            The Dataframe to construct the TableSchema from.

        Returns
        -------
        _from_dataframe: TableSchema
            The constructed TableSchema

        """

        names = dataframe.columns
        types = (ColumnType.from_numpy_dtype(dtype) for dtype in dataframe.dtypes)

        return TableSchema(dict(zip(names, types)))

    def get_column_names(self) -> list[str]:
        """
        Returns a list of all column names saved in this schema

        Returns
        -------
        column_names: list[str]
            the column names
        """
        return list(self._schema.keys())

    def __str__(self) -> str:
        """
        Returns a pretty print String for the TableSchema

        Returns
        -------
        output_string: str
            the pretty String
        """
        column_count = len(self._schema)
        output_string = f"TableSchema:\nColumn Count: {column_count}\nColumns:\n"
        for column_name, data_type in self._schema.items():
            output_string += f"    {column_name}: {data_type}\n"
        return output_string

    def __repr__(self) -> str:
        return self.__str__()
