from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from safeds.data.tabular.exceptions import UnknownColumnNameError
from safeds.data.tabular.typing._column_type import ColumnType

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


@dataclass
class Schema:
    """
    Store column names and corresponding data types for a `Table` or `Row`.

    Parameters
    ----------
    schema : dict[str, ColumnType]
        Map from column names to data types.
    """

    _schema: dict[str, ColumnType]

    @staticmethod
    def _from_pandas_dataframe(dataframe: pd.DataFrame) -> Schema:
        """
        Create a schema from a `pandas.DataFrame`.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe.

        Returns
        -------
        schema : Schema
            The schema.
        """
        names = dataframe.columns
        # noinspection PyProtectedMember
        types = (ColumnType._from_numpy_data_type(data_type) for data_type in dataframe.dtypes)

        return Schema(dict(zip(names, types, strict=True)))

    @staticmethod
    def _from_polars_dataframe(dataframe: pl.DataFrame) -> Schema:
        """
        Create a schema from a `polars.Dataframe`.

        Parameters
        ----------
        dataframe : pl.DataFrame
            The dataframe.

        Returns
        -------
        schema : Schema
            The schema.
        """
        names = dataframe.columns
        # noinspection PyProtectedMember
        types = (ColumnType._from_polars_data_type(data_type) for data_type in dataframe.dtypes)

        return Schema(dict(zip(names, types, strict=True)))

    def __init__(self, schema: dict[str, ColumnType]):
        self._schema = dict(schema)  # Defensive copy

    def __hash__(self) -> int:
        """
        Return a hash value for the schema.

        Returns
        -------
        hash : int
            The hash value.
        """
        column_names = self._schema.keys()
        column_types = map(repr, self._schema.values())
        return hash(tuple(zip(column_names, column_types, strict=True)))

    def __str__(self) -> str:
        """
        Return a user-friendly string representation of the schema.

        Returns
        -------
        string : str
            The string representation.
        """
        match len(self._schema):
            case 0:
                return "{}"
            case 1:
                return str(self._schema)
            case _:
                lines = (f"    {name!r}: {type_}" for name, type_ in self._schema.items())
                joined = ",\n".join(lines)
                return f"{{\n{joined}\n}}"

    @property
    def column_names(self) -> list[str]:
        """
        Return a list of all column names saved in this schema.

        Returns
        -------
        column_names : list[str]
            The column names.
        """
        return list(self._schema.keys())

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

    def get_column_type(self, column_name: str) -> ColumnType:
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
            If the specified column name does not exist.
        """
        if not self.has_column(column_name):
            raise UnknownColumnNameError([column_name])
        return self._schema[column_name]

    def _get_column_index(self, column_name: str) -> int:
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

        Raises
        ------
        ColumnNameError
            If the specified column name does not exist.
        """
        if not self.has_column(column_name):
            raise UnknownColumnNameError([column_name])

        return list(self._schema.keys()).index(column_name)
