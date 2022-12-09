from __future__ import annotations

import os.path
import typing
from pathlib import Path
from typing import Callable

import pandas as pd
from pandas import DataFrame, Series
from safe_ds.exceptions import (
    ColumnLengthMismatchError,
    ColumnNameDuplicateError,
    ColumnNameError,
    IndexOutOfBoundsError,
    SchemaMismatchError,
)

from ._column import Column
from ._column_type import ColumnType
from ._row import Row
from ._table_schema import TableSchema


# noinspection PyProtectedMember
class Table:
    def __init__(self, data: pd.DataFrame):
        self._data: pd.DataFrame = data
        self.schema: TableSchema = TableSchema(
            column_names=self._data.columns,
            data_types=list(
                map(ColumnType.from_numpy_dtype, self._data.dtypes.to_list())
            ),
        )

    def get_row(self, index: int) -> Row:
        """
        returns the row of the Table for a given Index
        Parameters
        ----------
        index : int

        Returns
        -------
        a Row of the Table
        Raises
        ------
        IndexOutOfBoundsError
            if the index doesn't exist
        """
        if len(self._data.index) - 1 < index or index < 0:
            raise IndexOutOfBoundsError(index)
        return Row(self._data.iloc[[index]].squeeze(), self.schema)

    @staticmethod
    def from_json(path: str) -> Table:
        """
        Reads data from a JSON file into a Table

        Parameters
        ----------
        path : str
            Path to the file as String

        Returns
        -------
        table : Table
            The Table read from the file

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist
        ValueError
            If the file could not be read
        """

        try:
            return Table(pd.read_json(path))
        except FileNotFoundError as exception:
            raise FileNotFoundError(f'File "{path}" does not exist') from exception
        except Exception as exception:
            raise ValueError(
                f'Could not read file from "{path}" as JSON'
            ) from exception

    @staticmethod
    def from_csv(path: str) -> Table:
        """
        Reads data from a CSV file into a Table.

        Parameters
        ----------
        path : str
            Path to the file as String

        Returns
        -------
        table : Table
            The Table read from the file

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist
        ValueError
            If the file could not be read
        """

        try:
            return Table(pd.read_csv(path))
        except FileNotFoundError as exception:
            raise FileNotFoundError(f'File "{path}" does not exist') from exception
        except Exception as exception:
            raise ValueError(f'Could not read file from "{path}" as CSV') from exception

    @staticmethod
    def from_rows(rows: list[Row]) -> Table:
        """
        Returns a table combined from a list of given rows.

        Parameters
        ----------
        rows : list[Row]
            Rows to be combined. Should have a matching schema.

        Returns
        -------
        table : Table
            The generated table.

        Raises
        ------
        SchemaMismatchError
            If one of the schemas of the rows does not match.
        """
        schema_compare: TableSchema = rows[0].schema
        row_array: list[Series] = []

        for row in rows:
            if schema_compare != row.schema:
                raise SchemaMismatchError()
            row_array.append(row._data)

        dataframe: DataFrame = pd.DataFrame(row_array)
        return Table(dataframe)

    @staticmethod
    def from_columns(columns: list[Column]) -> Table:
        """
        Returns a table combined from a list of given columns.

        Parameters
        ----------
        columns : list[Column]
            Columns to be combined. Each column should be the same size.

        Returns
        -------
        table : Table
            The generated table.

        Raises
        ------
        ColumnLengthMismatchError
            If at least one of the columns has a different length than at least one other column
        """
        dataframe: DataFrame = pd.DataFrame()

        for column in columns:
            if column._data.size != columns[0]._data.size:
                raise ColumnLengthMismatchError(
                    "\n".join(
                        [f"{column.name}: {column._data.size}" for column in columns]
                    )
                )
            dataframe[column.name] = column._data

        return Table(dataframe)

    def to_json(self, path_to_file: str) -> None:
        """
        Write the data from the table into a json file.
        If the file and/or the directories do not exist they will be created. If the file does already exist it will be overwritten.

        Parameters
        ----------
        path_to_file : The path as String to the output file.
        """
        Path(os.path.dirname(path_to_file)).mkdir(parents=True, exist_ok=True)
        self._data.to_json(path_to_file)

    def to_csv(self, path_to_file: str) -> None:
        """
        Write the data from the table into a csv file.
        If the file and/or the directories do not exist they will be created. If the file does already exist it will be overwritten.

        Parameters
        ----------
        path_to_file : The path as String to the output file.
        """
        Path(os.path.dirname(path_to_file)).mkdir(parents=True, exist_ok=True)
        self._data.to_csv(path_to_file, index=False)

    def rename_column(self, old_name: str, new_name: str) -> Table:
        """Rename a single column by providing the previous name and the future name of it.

        Parameters
        ----------
        old_name : str
            Old name of the target column
        new_name : str
            New name of the target column

        Returns
        -------
        table : Table
            The Table with the renamed column

        Raises
        ------
        ColumnNameError
            If the specified old target column name doesn't exist
        ColumnNameDuplicateError
            If the specified new target column name already exists
        """
        columns: list[str] = self._data.columns

        if old_name not in columns:
            raise ColumnNameError([old_name])
        if old_name == new_name:
            return self
        if new_name in columns:
            raise ColumnNameDuplicateError(new_name)

        return Table(self._data.rename(columns={old_name: new_name}))

    def get_column(self, column_name: str) -> Column:
        """Returns a new instance of Column with the data of the described column of the Table.

        Parameters
        ----------
        column_name : str
            The name of the column you want to get in return

        Returns
        -------
        column : Column
            A new instance of Column by the given name

        Raises
        ------
        ColumnNameError
            If the specified target column name doesn't exist
        """
        if column_name in self._data.columns:
            return Column(self._data[column_name].copy(deep=True), column_name)
        raise ColumnNameError([column_name])

    def drop_columns(self, column_names: list[str]) -> Table:
        """Returns a Table without the given columns

        Parameters
        ----------
        column_names : list[str]
            A List containing all columns to be dropped

        Returns
        -------
        table : Table
            A Table without the given columns

        Raises
        ------
        ColumnNameError
            If any of the given columns does not exist
        """
        invalid_columns = []
        for name in column_names:
            if name not in self._data.columns:
                invalid_columns.append(name)
        if len(invalid_columns) != 0:
            raise ColumnNameError(invalid_columns)
        transformed_data = self._data.drop(labels=column_names, axis="columns")
        return Table(transformed_data)

    def keep_columns(self, column_names: list[str]) -> Table:
        """Returns a Table with exactly the given columns

        Parameters
        ----------
        column_names : list[str]
            A List containing only the columns to be kept

        Returns
        -------
        table : Table
            A Table containing only the given columns

        Raises
        ------
        ColumnNameError
            If any of the given columns does not exist
        """
        invalid_columns = []
        for name in column_names:
            if name not in self._data.columns:
                invalid_columns.append(name)
        if len(invalid_columns) != 0:
            raise ColumnNameError(invalid_columns)
        transformed_data = self._data[column_names]
        return Table(transformed_data)

    def to_rows(self) -> list[Row]:
        """
        Returns a list of Rows from the current table.

        Returns
        -------
        rows : list[Row]
            List of Row objects
        """
        return [
            Row(series_row, self.schema) for (_, series_row) in self._data.iterrows()
        ]

    def filter_rows(self, query: Callable[[Row], bool]) -> Table:
        """Returns a Table with rows filtered by applied lambda function

        Parameters
        ----------
        query : lambda function
            A lambda function that is applied to all rows

        Returns
        -------
        table : Table
            A Table containing only the rows filtered by the query lambda function

        Raises
        ------
        TypeError
           If the entered query is not a lambda function
        """

        rows: list[Row] = [row for row in self.to_rows() if query(row)]
        result_table: Table = self.from_rows(rows)
        return Table(result_table._data.reset_index(drop=True))

    def count_rows(self) -> int:
        """
        Returns the number of rows in the table

        Returns
        -------
        count : int
            Number of rows
        """
        return self._data.shape[0]

    def to_columns(self) -> list[Column]:
        """
        Returns a list of Columns from the current table.

        Returns
        -------
        columns : list[Columns]
            List of Columns objects
        """
        return [self.get_column(name) for name in self._data.columns]

    def drop_duplicate_rows(self) -> Table:
        """
        Returns a copy of the Table with every duplicate row removed.

        Returns
        -------
        result: Table
            The table with the duplicate rows removed

        """
        return Table(self._data.drop_duplicates(ignore_index=True))

    def __eq__(self, other: typing.Any) -> bool:
        if not isinstance(other, Table):
            return NotImplemented
        if self is other:
            return True
        return self._data.equals(other._data)

    def __hash__(self) -> int:
        return hash(self._data)
