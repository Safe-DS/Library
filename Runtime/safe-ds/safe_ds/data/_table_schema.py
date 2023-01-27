from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from safe_ds.exceptions import UnknownColumnNameError

from ._column_type import ColumnType


@dataclass
class TableSchema:
    """
    Store column names and corresponding data types for a table.

    Parameters
    ----------
    schema : dict[str, ColumnType]
        Map from column names to data types.
    """

    _schema: dict[str, ColumnType]

    def __init__(self, schema: dict[str, ColumnType]):
        self._schema = dict(schema)  # Defensive copy

    def has_column(self, column_name: str) -> bool:
        """
        Return whether the schema contains a given column.

        Parameters
        ----------
        column_name : str
            The name of the column.

        Returns
        -------
        contains : bool
            True if the schema contains the column.
        """
        return column_name in self._schema

    def get_type_of_column(self, column_name: str) -> ColumnType:
        """
        Return the type of the given column.

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
        if not self.has_column(column_name):
            raise UnknownColumnNameError([column_name])
        return self._schema[column_name]

    def _get_column_index_by_name(self, column_name: str) -> int:
        """
         Return the index of the column with specified column name.

         Parameters
         ----------
         column_name : str
             The name of the column.

         Returns
         -------
        index : int
             The index of the column.
        """

        return list(self._schema.keys()).index(column_name)

    @staticmethod
    def _from_dataframe(dataframe: pd.DataFrame) -> TableSchema:
        """
        Construct a TableSchema from a Dataframe. This function is not supposed to be exposed to the user.

        Parameters
        ----------
        dataframe : pd.Dataframe
            The Dataframe used to construct the TableSchema.

        Returns
        -------
        _from_dataframe: TableSchema
            The constructed TableSchema.

        """

        names = dataframe.columns
        types = (ColumnType.from_numpy_dtype(dtype) for dtype in dataframe.dtypes)

        return TableSchema(dict(zip(names, types)))

    def get_column_names(self) -> list[str]:
        """
        Return a list of all column names saved in this schema.

        Returns
        -------
        column_names : list[str]
            The column names.
        """
        return list(self._schema.keys())

    def __str__(self) -> str:
        """
        Return a print-string for the TableSchema.

        Returns
        -------
        output_string : str
            The string.
        """
        column_count = len(self._schema)
        output_string = f"TableSchema:\nColumn Count: {column_count}\nColumns:\n"
        for column_name, data_type in self._schema.items():
            output_string += f"    {column_name}: {data_type}\n"
        return output_string

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, TableSchema):
            return NotImplemented
        if self is o:
            return True
        return self._schema == o._schema
