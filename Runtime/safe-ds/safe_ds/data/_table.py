from __future__ import annotations

import os.path
import typing
from pathlib import Path
from typing import Callable

import pandas as pd
from pandas import DataFrame, Series
from safe_ds.exceptions import (
    ColumnLengthMismatchError,
    ColumnSizeError,
    DuplicateColumnNameError,
    IndexOutOfBoundsError,
    SchemaMismatchError,
    UnknownColumnNameError,
)

from ._column import Column
from ._row import Row
from ._table_schema import TableSchema


# noinspection PyProtectedMember
class Table:
    def __init__(self, data: typing.Iterable):
        self._data: pd.Dataframe = (
            data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        )
        self.schema: TableSchema = TableSchema._from_dataframe(self._data)

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
        DuplicateColumnNameError
            If the specified new target column name already exists
        """
        columns: list[str] = self._data.columns

        if old_name not in columns:
            raise UnknownColumnNameError([old_name])
        if old_name == new_name:
            return self
        if new_name in columns:
            raise DuplicateColumnNameError(new_name)

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
        UnknownColumnNameError
            If the specified target column name doesn't exist
        """
        if column_name in self._data.columns:
            return Column(self._data[column_name].copy(deep=True), column_name)

        raise UnknownColumnNameError([column_name])

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
            raise UnknownColumnNameError(invalid_columns)
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
            raise UnknownColumnNameError(invalid_columns)
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

    def count_columns(self) -> int:
        """
        Returns the number of columns in the table

        Returns
        -------
        count : int
            Number of columns
        """
        return self._data.shape[1]

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

    def replace_column(self, old_column_name: str, new_column: Column) -> Table:
        """
        Returns a copy of the Table with the specified old column replaced by a new column. Keeps the order of columns.

        Parameters
        ----------
        old_column_name: str
            Name of the old column, to be replaced

        new_column: Column
            New column, to replace the old column

        Returns
        -------
        result: Table
            Table where the old column is replaced by the new column

        Raises
        ------
        UnknownColumnNameError
            If the old column does not exist

        DuplicateColumnNameError
            If the new column already exists and the existing column is not affected by the replacement

        ColumnSizeError
            If the size of the column does not match the amount of rows
        """
        columns = self._data.columns

        if old_column_name not in columns:
            raise UnknownColumnNameError([old_column_name])

        if new_column.name in columns and new_column.name != old_column_name:
            raise DuplicateColumnNameError(new_column.name)

        if self.count_rows() != new_column._data.size:
            raise ColumnSizeError(str(self.count_rows()), str(new_column._data.size))

        if old_column_name != new_column.name:
            result = self.rename_column(old_column_name, new_column.name)._data
        else:
            result = self._data.copy()

        result[new_column.name] = new_column._data
        return Table(result)

    def add_column(self, column: Column) -> Table:
        """
        Returns the original table with the provided column attached at the end.

        Returns
        -------
        result: Table
            The table with the column attached

        Raises
        ------
        DuplicateColumnNameError
            If the new column already exists

        ColumnSizeError
            If the size of the column does not match the amount of rows

        """
        if column.name in self._data.columns:
            raise DuplicateColumnNameError(column.name)

        if column._data.size != self.count_rows():
            raise ColumnSizeError(str(self.count_rows()), str(column._data.size))

        result = self._data.copy()
        result[column.name] = column._data
        return Table(result)

    def add_row(self, row: Row) -> Table:
        """
        Add a row to an existing table

        Parameters
        ----------
        row: Row
            the row you want to add

        Returns
        -------
        table: Table
            a new table with the added row at the end

        """
        if self.schema != row.schema:
            raise SchemaMismatchError()
        df = row._data.to_frame().T
        df.columns = list(self.schema._schema.keys())
        return Table(pd.concat([self._data, df], ignore_index=True))

    def has_column(self, column_name: str) -> bool:
        """
        Returns if the table contains a given column

        Parameters
        ----------
        column_name : str
            The name of the column

        Returns
        -------
        contains: bool
            If it contains the column
        """
        return self.schema.has_column(column_name)

    def list_columns_with_missing_values(self) -> list[Column]:
        """
        Returns a list of all the columns, that have at least one missing value or an empty list, if there are none.

        Returns
        -------
        columns_with_missing_values: list[Column]
            The list of columns with missing values
        """
        columns = self.to_columns()
        columns_with_missing_values = []
        for column in columns:
            if column.has_missing_values():
                columns_with_missing_values.append(column)
        return columns_with_missing_values

    def list_columns_with_non_numerical_values(self) -> list[Column]:
        """
        Get a list of Columns only containing non-numerical values

        Returns
        -------
        cols: list[Column]
            the list with only non-numerical Columns
        """
        cols = []
        for column_name, data_type in self.schema._schema.items():
            if not data_type.is_numeric():
                cols.append(self.get_column(column_name))
        return cols

    def __eq__(self, other: typing.Any) -> bool:
        if not isinstance(other, Table):
            return NotImplemented
        if self is other:
            return True
        return (
            self._data.reset_index(drop=True, inplace=False).equals(
                other._data.reset_index(drop=True, inplace=False)
            )
            and self.schema == other.schema
        )

    def __hash__(self) -> int:
        return hash(self._data)

    def transform_column(
        self, name: str, transformer: Callable[[Row], typing.Any]
    ) -> Table:
        """
        Transform provided column by calling provided transformer

        Returns
        -------
        result: Table
            The table with the transformed column

        Raises
        ------
        UnknownColumnNameError
            If the old column does not exist

        """
        if self.has_column(name):
            items: list = [transformer(item) for item in self.to_rows()]
            result: Column = Column(pd.Series(items), name)
            return self.replace_column(name, result)
        raise UnknownColumnNameError([name])
