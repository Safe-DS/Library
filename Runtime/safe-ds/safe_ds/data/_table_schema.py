from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from safe_ds.exceptions import ColumnNameError


@dataclass
class TableSchema:
    """Stores column names and corresponding data types for a table

    Parameters
    ----------
    column_names: list[str]
        Column names as an array
    data_types: list[numpy.dtype]
        Dataypes as an array using the numpy dtpye class

    """

    _schema: OrderedDict

    def __init__(self, column_names: list[str], data_types: list[np.dtype]):
        self._schema = OrderedDict()
        for column_name, data_type in zip(column_names, data_types):
            self._schema[column_name] = data_type

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

    def get_type_of_column(self, column_name: str) -> np.dtype:
        """
        Returns the type of the given column

        Parameters
        ----------
        column_name : str
            The name of the column you want the type of

        Returns
        -------
        type: np.dtype
            The type of the column

        Raises
        ------
        ColumnNameError
            If the specified target column name doesn't exist
        """
        if not self.has_column(column_name):
            raise ColumnNameError([column_name])
        return self._schema[column_name]

    def _get_column_index_by_name(self, column_name: str):
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
            output_string += f"    {column_name}, {data_type}\n"
        return output_string
