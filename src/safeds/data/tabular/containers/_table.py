from __future__ import annotations

import functools
import io
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from scipy import stats

from safeds.data.image.containers import Image
from safeds.data.image.typing import ImageFormat
from safeds.data.tabular.typing import ColumnType, Schema
from safeds.exceptions import (
    ColumnLengthMismatchError,
    ColumnSizeError,
    DuplicateColumnNameError,
    IndexOutOfBoundsError,
    NonNumericColumnError,
    SchemaMismatchError,
    UnknownColumnNameError,
    WrongFileExtensionError,
)

from ._column import Column
from ._row import Row

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from safeds.data.tabular.transformation import InvertibleTableTransformer, TableTransformer

    from ._tagged_table import TaggedTable


# noinspection PyProtectedMember
class Table:
    """
    A table is a two-dimensional collection of data. It can either be seen as a list of rows or as a list of columns.

    To create a `Table` call the constructor or use one of the following static methods:

    | Method                                                                       | Description                            |
    | ---------------------------------------------------------------------------- | -------------------------------------- |
    | [from_csv_file][safeds.data.tabular.containers._table.Table.from_csv_file]   | Create a table from a CSV file.        |
    | [from_json_file][safeds.data.tabular.containers._table.Table.from_json_file] | Create a table from a JSON file.       |
    | [from_dict][safeds.data.tabular.containers._table.Table.from_dict]           | Create a table from a dictionary.      |
    | [from_columns][safeds.data.tabular.containers._table.Table.from_columns]     | Create a table from a list of columns. |
    | [from_rows][safeds.data.tabular.containers._table.Table.from_rows]           | Create a table from a list of rows.    |

    Parameters
    ----------
    data : Mapping[str, Sequence[Any]] | None
        The data. If None, an empty table is created.

    Raises
    ------
    ColumnLengthMismatchError
        If columns have different lengths.

    Examples
    --------
    >>> from safeds.data.tabular.containers import Table
    >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def from_csv_file(path: str | Path) -> Table:
        """
        Read data from a CSV file into a table.

        Parameters
        ----------
        path : str | Path
            The path to the CSV file.

        Returns
        -------
        table : Table
            The table created from the CSV file.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        WrongFileExtensionError
            If the file is not a csv file.
        """
        path = Path(path)
        if path.suffix != ".csv":
            raise WrongFileExtensionError(path, ".csv")
        try:
            return Table._from_pandas_dataframe(pd.read_csv(path))
        except FileNotFoundError as exception:
            raise FileNotFoundError(f'File "{path}" does not exist') from exception

    @staticmethod
    def from_excel_file(path: str | Path) -> Table:
        """
        Read data from an Excel file into a table.

        Valid file extensions are `.xls`, '.xlsx', `.xlsm`, `.xlsb`, `.odf`, `.ods` and `.odt`.

        Parameters
        ----------
        path : str | Path
            The path to the Excel file.

        Returns
        -------
        table : Table
            The table created from the Excel file.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        WrongFileExtensionError
            If the file is not an Excel file.
        """
        path = Path(path)
        excel_extensions = [".xls", ".xlsx", ".xlsm", ".xlsb", ".odf", ".ods", ".odt"]
        if path.suffix not in excel_extensions:
            raise WrongFileExtensionError(path, excel_extensions)
        try:
            return Table._from_pandas_dataframe(
                pd.read_excel(path, engine="openpyxl", usecols=lambda colname: "Unnamed" not in colname),
            )
        except FileNotFoundError as exception:
            raise FileNotFoundError(f'File "{path}" does not exist') from exception

    @staticmethod
    def from_json_file(path: str | Path) -> Table:
        """
        Read data from a JSON file into a table.

        Parameters
        ----------
        path : str | Path
            The path to the JSON file.

        Returns
        -------
        table : Table
            The table created from the JSON file.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        WrongFileExtensionError
            If the file is not a JSON file.
        """
        path = Path(path)
        if path.suffix != ".json":
            raise WrongFileExtensionError(path, ".json")
        try:
            return Table._from_pandas_dataframe(pd.read_json(path))
        except FileNotFoundError as exception:
            raise FileNotFoundError(f'File "{path}" does not exist') from exception

    @staticmethod
    def from_dict(data: dict[str, list[Any]]) -> Table:
        """
        Create a table from a dictionary that maps column names to column values.

        Parameters
        ----------
        data : dict[str, list[Any]]
            The data.

        Returns
        -------
        table : Table
            The generated table.

        Raises
        ------
        ColumnLengthMismatchError
            If columns have different lengths.
        """
        return Table(data)

    @staticmethod
    def from_columns(columns: list[Column]) -> Table:
        """
        Return a table created from a list of columns.

        Parameters
        ----------
        columns : list[Column]
            The columns to be combined. They need to have the same size.

        Returns
        -------
        table : Table
            The generated table.

        Raises
        ------
        ColumnLengthMismatchError
            If any of the column sizes does not match with the others.
        DuplicateColumnNameError
            If multiple columns have the same name.
        """
        dataframe: DataFrame = pd.DataFrame()
        column_names = []

        for column in columns:
            if column._data.size != columns[0]._data.size:
                raise ColumnLengthMismatchError(
                    "\n".join(f"{column.name}: {column._data.size}" for column in columns),
                )
            if column.name in column_names:
                raise DuplicateColumnNameError(column.name)
            column_names.append(column.name)
            dataframe[column.name] = column._data

        return Table._from_pandas_dataframe(dataframe)

    @staticmethod
    def from_rows(rows: list[Row]) -> Table:
        """
        Return a table created from a list of rows.

        Parameters
        ----------
        rows : list[Row]
            The rows to be combined. They need to have a matching schema.

        Returns
        -------
        table : Table
            The generated table.

        Raises
        ------
        SchemaMismatchError
            If any of the row schemas does not match with the others.
        """
        if len(rows) == 0:
            return Table._from_pandas_dataframe(pd.DataFrame())

        schema_compare: Schema = rows[0]._schema
        row_array: list[pd.DataFrame] = []

        for row in rows:
            if schema_compare != row._schema:
                raise SchemaMismatchError
            row_array.append(row._data)

        dataframe: DataFrame = pd.concat(row_array, ignore_index=True)
        dataframe.columns = schema_compare.column_names
        return Table._from_pandas_dataframe(dataframe)

    @staticmethod
    def _from_pandas_dataframe(data: pd.DataFrame, schema: Schema | None = None) -> Table:
        """
        Create a table from a `pandas.DataFrame`.

        Parameters
        ----------
        data : pd.DataFrame
            The data.
        schema : Schema | None
            The schema. If None, the schema is inferred from the data.

        Returns
        -------
        table : Table
            The created table.

        Examples
        --------
        >>> import pandas as pd
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table._from_pandas_dataframe(pd.DataFrame({"a": [1], "b": [2]}))
        """
        data = data.reset_index(drop=True)

        result = object.__new__(Table)
        result._data = data

        if schema is None:
            # noinspection PyProtectedMember
            result._schema = Schema._from_pandas_dataframe(data)
        else:
            result._schema = schema
            if result._data.empty:
                result._data = pd.DataFrame(columns=schema.column_names)

        return result

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, data: Mapping[str, Sequence[Any]] | None = None) -> None:
        """
        Create a table from a mapping of column names to their values.

        Parameters
        ----------
        data : Mapping[str, Sequence[Any]] | None
            The data. If None, an empty table is created.

        Raises
        ------
        ColumnLengthMismatchError
            If columns have different lengths.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        """
        if data is None:
            data = {}

        # Validation
        expected_length: int | None = None
        for column_values in data.values():
            if expected_length is None:
                expected_length = len(column_values)
            elif len(column_values) != expected_length:
                raise ColumnLengthMismatchError(
                    "\n".join(f"{column_name}: {len(column_values)}" for column_name, column_values in data.items()),
                )

        # Implementation
        self._data: pd.DataFrame = pd.DataFrame(data)
        self._data = self._data.reset_index(drop=True)
        self._schema: Schema = Schema._from_pandas_dataframe(self._data)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Table):
            return NotImplemented
        if self is other:
            return True
        if self.number_of_rows == 0 and other.number_of_rows == 0:
            return self.column_names == other.column_names
        table1 = self.sort_columns()
        table2 = other.sort_columns()
        return table1._schema == table2._schema and table1._data.equals(table2._data)

    def __repr__(self) -> str:
        tmp = self._data.copy(deep=True)
        tmp.columns = self.column_names
        return tmp.__repr__()

    def __str__(self) -> str:
        tmp = self._data.copy(deep=True)
        tmp.columns = self.column_names
        return tmp.__str__()

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def column_names(self) -> list[str]:
        """
        Return a list of all column names in this table.

        Alias for self.schema.column_names -> list[str].

        Returns
        -------
        column_names : list[str]
            The list of the column names.
        """
        return self._schema.column_names

    @property
    def number_of_columns(self) -> int:
        """
        Return the number of columns.

        Returns
        -------
        number_of_columns : int
            The number of columns.
        """
        return self._data.shape[1]

    @property
    def number_of_rows(self) -> int:
        """
        Return the number of rows.

        Returns
        -------
        number_of_rows : int
            The number of rows.
        """
        return self._data.shape[0]

    @property
    def schema(self) -> Schema:
        """
        Return the schema of the table.

        Returns
        -------
        schema : Schema
            The schema.
        """
        return self._schema

    # ------------------------------------------------------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------------------------------------------------------

    def get_column(self, column_name: str) -> Column:
        """
        Return a column with the data of the specified column.

        Parameters
        ----------
        column_name : str
            The name of the column.

        Returns
        -------
        column : Column
            The column.

        Raises
        ------
        UnknownColumnNameError
            If the specified target column name does not exist.
        """
        if not self.has_column(column_name):
            raise UnknownColumnNameError([column_name])

        return Column._from_pandas_series(
            self._data[column_name],
            self.get_column_type(column_name),
        )

    def has_column(self, column_name: str) -> bool:
        """
        Return whether the table contains a given column.

        Alias for self.schema.hasColumn(column_name: str) -> bool.

        Parameters
        ----------
        column_name : str
            The name of the column.

        Returns
        -------
        contains : bool
            True if the column exists.
        """
        return self._schema.has_column(column_name)

    def get_column_type(self, column_name: str) -> ColumnType:
        """
        Return the type of the given column.

        Alias for self.schema.get_type_of_column(column_name: str) -> ColumnType.

        Parameters
        ----------
        column_name : str
            The name of the column to be queried.

        Returns
        -------
        type : ColumnType
            The type of the column.

        Raises
        ------
        UnknownColumnNameError
            If the specified target column name does not exist.
        """
        return self._schema.get_column_type(column_name)

    def get_row(self, index: int) -> Row:
        """
        Return the row at a specified index.

        Parameters
        ----------
        index : int
            The index.

        Returns
        -------
        row : Row
            The row of the table at the index.

        Raises
        ------
        IndexOutOfBoundsError
            If no row at the specified index exists in this table.
        """
        if len(self._data.index) - 1 < index or index < 0:
            raise IndexOutOfBoundsError(index)

        return Row._from_pandas_dataframe(self._data.iloc[[index]], self._schema)

    # ------------------------------------------------------------------------------------------------------------------
    # Information
    # ------------------------------------------------------------------------------------------------------------------

    def summary(self) -> Table:
        """
        Return a table with a number of statistical key values.

        This table is not modified.

        Returns
        -------
        result : Table
            The table with statistics.
        """
        columns = self.to_columns()
        result = pd.DataFrame()
        statistics = {}

        for column in columns:
            statistics = {
                "maximum": column.maximum,
                "minimum": column.minimum,
                "mean": column.mean,
                "mode": column.mode,
                "median": column.median,
                "sum": column.sum,
                "variance": column.variance,
                "standard deviation": column.standard_deviation,
                "idness": column.idness,
                "stability": column.stability,
            }
            values = []

            for function in statistics.values():
                try:
                    values.append(str(function()))
                except NonNumericColumnError:
                    values.append("-")

            result = pd.concat([result, pd.DataFrame(values)], axis=1)

        result = pd.concat([pd.DataFrame(list(statistics.keys())), result], axis=1)
        result.columns = ["metrics", *self.column_names]

        return Table._from_pandas_dataframe(result)

    # ------------------------------------------------------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------------------------------------------------------

    def add_column(self, column: Column) -> Table:
        """
        Return the original table with the provided column attached at the end.

        This table is not modified.

        Returns
        -------
        result : Table
            The table with the column attached.

        Raises
        ------
        DuplicateColumnNameError
            If the new column already exists.

        ColumnSizeError
            If the size of the column does not match the amount of rows.

        """
        if self.has_column(column.name):
            raise DuplicateColumnNameError(column.name)

        if column._data.size != self.number_of_rows:
            raise ColumnSizeError(str(self.number_of_rows), str(column._data.size))

        result = self._data.copy()
        result.columns = self._schema.column_names
        result[column.name] = column._data
        return Table._from_pandas_dataframe(result)

    def add_columns(self, columns: list[Column] | Table) -> Table:
        """
        Add multiple columns to the table.

        This table is not modified.

        Parameters
        ----------
        columns : list[Column] or Table
            The columns to be added.

        Returns
        -------
        result: Table
            A new table combining the original table and the given columns.

        Raises
        ------
        ColumnSizeError
            If at least one of the column sizes from the provided column list does not match the table.
        DuplicateColumnNameError
            If at least one column name from the provided column list already exists in the table.
        """
        if isinstance(columns, Table):
            columns = columns.to_columns()
        result = self._data.copy()
        result.columns = self._schema.column_names
        for column in columns:
            if column.name in result.columns:
                raise DuplicateColumnNameError(column.name)

            if column._data.size != self.number_of_rows:
                raise ColumnSizeError(str(self.number_of_rows), str(column._data.size))

            result[column.name] = column._data
        return Table._from_pandas_dataframe(result)

    def add_row(self, row: Row) -> Table:
        """
        Add a row to the table.

        This table is not modified.

        Parameters
        ----------
        row : Row
            The row to be added.

        Returns
        -------
        table : Table
            A new table with the added row at the end.

        Raises
        ------
        SchemaMismatchError
            If the schema of the row does not match the table schema.
        """
        if self._schema != row.schema:
            raise SchemaMismatchError

        new_df = pd.concat([self._data, row._data]).infer_objects()
        new_df.columns = self.column_names
        return Table._from_pandas_dataframe(new_df)

    def add_rows(self, rows: list[Row] | Table) -> Table:
        """
        Add multiple rows to a table.

        This table is not modified.

        Parameters
        ----------
        rows : list[Row] or Table
            The rows to be added.

        Returns
        -------
        result : Table
            A new table which combines the original table and the given rows.

        Raises
        ------
        SchemaMismatchError
            If the schema of on of the row does not match the table schema.
        """
        if isinstance(rows, Table):
            rows = rows.to_rows()
        result = self._data
        for row in rows:
            if self._schema != row.schema:
                raise SchemaMismatchError

        row_frames = (row._data for row in rows)

        result = pd.concat([result, *row_frames]).infer_objects()
        result.columns = self.column_names
        return Table._from_pandas_dataframe(result)

    def filter_rows(self, query: Callable[[Row], bool]) -> Table:
        """
        Return a table with rows filtered by Callable (e.g. lambda function).

        This table is not modified.

        Parameters
        ----------
        query : lambda function
            A Callable that is applied to all rows.

        Returns
        -------
        table : Table
            A table containing only the rows filtered by the query.
        """
        rows: list[Row] = [row for row in self.to_rows() if query(row)]
        if len(rows) == 0:
            result_table = Table._from_pandas_dataframe(pd.DataFrame(), self._schema)
        else:
            result_table = self.from_rows(rows)
        return result_table

    def keep_only_columns(self, column_names: list[str]) -> Table:
        """
        Return a table with only the given column(s).

        This table is not modified.

        Parameters
        ----------
        column_names : list[str]
            A list containing only the columns to be kept.

        Returns
        -------
        table : Table
            A table containing only the given column(s).

        Raises
        ------
        UnknownColumnNameError
            If any of the given columns does not exist.
        """
        invalid_columns = []
        for name in column_names:
            if not self._schema.has_column(name):
                invalid_columns.append(name)
        if len(invalid_columns) != 0:
            raise UnknownColumnNameError(invalid_columns)

        transformed_data = self._data[column_names]
        transformed_data.columns = column_names
        return Table._from_pandas_dataframe(transformed_data)

    def remove_columns(self, column_names: list[str]) -> Table:
        """
        Return a table without the given column(s).

        This table is not modified.

        Parameters
        ----------
        column_names : list[str]
            A list containing all columns to be dropped.

        Returns
        -------
        table : Table
            A table without the given columns.

        Raises
        ------
        UnknownColumnNameError
            If any of the given columns does not exist.
        """
        invalid_columns = []
        for name in column_names:
            if not self._schema.has_column(name):
                invalid_columns.append(name)
        if len(invalid_columns) != 0:
            raise UnknownColumnNameError(invalid_columns)

        transformed_data = self._data.drop(labels=column_names, axis="columns")
        transformed_data.columns = [name for name in self._schema.column_names if name not in column_names]
        return Table._from_pandas_dataframe(transformed_data)

    def remove_columns_with_missing_values(self) -> Table:
        """
        Return a table without the columns that contain missing values.

        This table is not modified.

        Returns
        -------
        table : Table
            A table without the columns that contain missing values.
        """
        return Table.from_columns([column for column in self.to_columns() if not column.has_missing_values()])

    def remove_columns_with_non_numerical_values(self) -> Table:
        """
        Return a table without the columns that contain non-numerical values.

        This table is not modified.

        Returns
        -------
        table : Table
            A table without the columns that contain non-numerical values.

        """
        return Table.from_columns([column for column in self.to_columns() if column.type.is_numeric()])

    def remove_duplicate_rows(self) -> Table:
        """
        Return a copy of the table with every duplicate row removed.

        This table is not modified.

        Returns
        -------
        result : Table
            The table with the duplicate rows removed.
        """
        result = self._data.drop_duplicates(ignore_index=True)
        result.columns = self._schema.column_names
        return Table._from_pandas_dataframe(result)

    def remove_rows_with_missing_values(self) -> Table:
        """
        Return a table without the rows that contain missing values.

        This table is not modified.

        Returns
        -------
        table : Table
            A table without the rows that contain missing values.
        """
        result = self._data.copy(deep=True)
        result = result.dropna(axis="index")
        return Table._from_pandas_dataframe(result, self._schema)

    def remove_rows_with_outliers(self) -> Table:
        """
        Remove all rows from the table that contain at least one outlier.

        We define an outlier as a value that has a distance of more than 3 standard deviations from the column mean.
        Missing values are not considered outliers. They are also ignored during the calculation of the standard
        deviation.

        This table is not modified.

        Returns
        -------
        new_table : Table
            A new table without rows containing outliers.
        """
        copy = self._data.copy(deep=True)

        table_without_nonnumericals = self.remove_columns_with_non_numerical_values()
        z_scores = np.absolute(stats.zscore(table_without_nonnumericals._data, nan_policy="omit"))
        filter_ = ((z_scores < 3) | np.isnan(z_scores)).all(axis=1)

        return Table._from_pandas_dataframe(copy[filter_], self._schema)

    def rename_column(self, old_name: str, new_name: str) -> Table:
        """
        Rename a single column.

        This table is not modified.

        Parameters
        ----------
        old_name : str
            The old name of the target column
        new_name : str
            The new name of the target column

        Returns
        -------
        table : Table
            The Table with the renamed column.

        Raises
        ------
        UnknownColumnNameError
            If the specified old target column name does not exist.
        DuplicateColumnNameError
            If the specified new target column name already exists.
        """
        if old_name not in self._schema.column_names:
            raise UnknownColumnNameError([old_name])
        if old_name == new_name:
            return self
        if new_name in self._schema.column_names:
            raise DuplicateColumnNameError(new_name)

        new_df = self._data.copy()
        new_df.columns = self._schema.column_names
        return Table._from_pandas_dataframe(new_df.rename(columns={old_name: new_name}))

    def replace_column(self, old_column_name: str, new_column: Column) -> Table:
        """
        Return a copy of the table with the specified old column replaced by a new column. Keeps the order of columns.

        This table is not modified.

        Parameters
        ----------
        old_column_name : str
            The name of the column to be replaced.

        new_column : Column
            The new column replacing the old column.

        Returns
        -------
        result : Table
            A table with the old column replaced by the new column.

        Raises
        ------
        UnknownColumnNameError
            If the old column does not exist.

        DuplicateColumnNameError
            If the new column already exists and the existing column is not affected by the replacement.

        ColumnSizeError
            If the size of the column does not match the amount of rows.
        """
        if old_column_name not in self._schema.column_names:
            raise UnknownColumnNameError([old_column_name])

        if new_column.name in self._schema.column_names and new_column.name != old_column_name:
            raise DuplicateColumnNameError(new_column.name)

        if self.number_of_rows != new_column._data.size:
            raise ColumnSizeError(str(self.number_of_rows), str(new_column._data.size))

        if old_column_name != new_column.name:
            renamed_table = self.rename_column(old_column_name, new_column.name)
            result = renamed_table._data
            result.columns = renamed_table._schema.column_names
        else:
            result = self._data.copy()
            result.columns = self._schema.column_names

        result[new_column.name] = new_column._data
        return Table._from_pandas_dataframe(result)

    def shuffle_rows(self) -> Table:
        """
        Shuffle the table randomly.

        This table is not modified.

        Returns
        -------
        result : Table
            The shuffled Table.

        """
        new_df = self._data.sample(frac=1.0)
        new_df.columns = self._schema.column_names
        return Table._from_pandas_dataframe(new_df)

    def slice_rows(
        self,
        start: int | None = None,
        end: int | None = None,
        step: int = 1,
    ) -> Table:
        """
        Slice a part of the table into a new table.

        This table is not modified.

        Parameters
        ----------
        start : int
            The first index of the range to be copied into a new table, None by default.
        end : int
            The last index of the range to be copied into a new table, None by default.
        step : int
            The step size used to iterate through the table, 1 by default.

        Returns
        -------
        result : Table
            The resulting table.

        Raises
        ------
        IndexOutOfBoundsError
            If the index is out of bounds.
        """
        if start is None:
            start = 0

        if end is None:
            end = self.number_of_rows

        if end < start:
            raise IndexOutOfBoundsError(slice(start, end))
        if start < 0 or end < 0 or start > self.number_of_rows or end > self.number_of_rows:
            raise IndexOutOfBoundsError(start if start < 0 or start > self.number_of_rows else end)

        new_df = self._data.iloc[start:end:step]
        new_df.columns = self._schema.column_names
        return Table._from_pandas_dataframe(new_df)

    def sort_columns(
        self,
        comparator: Callable[[Column, Column], int] = lambda col1, col2: (col1.name > col2.name)
        - (col1.name < col2.name),
    ) -> Table:
        """
        Sort the columns of a `Table` with the given comparator and return a new `Table`.

        The original table is not modified. The comparator is a function that takes two columns `col1` and `col2` and
        returns an integer:

        * If `col1` should be ordered before `col2`, the function should return a negative number.
        * If `col1` should be ordered after `col2`, the function should return a positive number.
        * If the original order of `col1` and `col2` should be kept, the function should return 0.

        If no comparator is given, the columns will be sorted alphabetically by their name.

        This table is not modified.

        Parameters
        ----------
        comparator : Callable[[Column, Column], int]
            The function used to compare two columns.

        Returns
        -------
        new_table : Table
            A new table with sorted columns.
        """
        columns = self.to_columns()
        columns.sort(key=functools.cmp_to_key(comparator))
        return Table.from_columns(columns)

    def sort_rows(self, comparator: Callable[[Row, Row], int]) -> Table:
        """
        Sort the rows of a `Table` with the given comparator and return a new `Table`.

        The original table is not modified. The comparator is a function that takes two rows `row1` and `row2` and
        returns an integer:

        * If `row1` should be ordered before `row2`, the function should return a negative number.
        * If `row1` should be ordered after `row2`, the function should return a positive number.
        * If the original order of `row1` and `row2` should be kept, the function should return 0.

        This table is not modified.

        Parameters
        ----------
        comparator : Callable[[Row, Row], int]
            The function used to compare two rows.

        Returns
        -------
        new_table : Table
            A new table with sorted rows.
        """
        rows = self.to_rows()
        rows.sort(key=functools.cmp_to_key(comparator))
        return Table.from_rows(rows)

    def split(self, percentage_in_first: float) -> tuple[Table, Table]:
        """
        Split the table into two new tables.

        This table is not modified.

        Parameters
        ----------
        percentage_in_first : float
            The desired size of the first table in percentage to the given table.

        Returns
        -------
        result : (Table, Table)
            A tuple containing the two resulting tables. The first table has the specified size, the second table
            contains the rest of the data.

        Raises
        ------
        ValueError:
            if the 'percentage_in_first' is not between 0 and 1
        """
        if percentage_in_first < 0 or percentage_in_first > 1:
            raise ValueError("The given percentage is not between 0 and 1")
        return (
            self.slice_rows(0, round(percentage_in_first * self.number_of_rows)),
            self.slice_rows(round(percentage_in_first * self.number_of_rows)),
        )

    def tag_columns(self, target_name: str, feature_names: list[str] | None = None) -> TaggedTable:
        """
        Mark the columns of the table as target column or feature columns. The original table is not modified.

        This table is not modified.

        Parameters
        ----------
        target_name : str
            Name of the target column.
        feature_names : list[str] | None
            Names of the feature columns. If None, all columns except the target column are used.

        Returns
        -------
        tagged_table : TaggedTable
            A new tagged table with the given target and feature names.

        Raises
        ------
        ValueError
            If the target column is also a feature column.
        ValueError
            If no feature columns are specified.
        """
        from ._tagged_table import TaggedTable

        return TaggedTable._from_table(self, target_name, feature_names)

    def transform_column(self, name: str, transformer: Callable[[Row], Any]) -> Table:
        """
        Transform provided column by calling provided transformer.

        This table is not modified.

        Returns
        -------
        result : Table
            The table with the transformed column.

        Raises
        ------
        UnknownColumnNameError
            If the column does not exist.

        """
        if self.has_column(name):
            items: list = [transformer(item) for item in self.to_rows()]
            result: Column = Column(name, items)
            return self.replace_column(name, result)
        raise UnknownColumnNameError([name])

    def transform_table(self, transformer: TableTransformer) -> Table:
        """
        Apply a learned transformation onto this table.

        This table is not modified.

        Parameters
        ----------
        transformer : TableTransformer
            The transformer which transforms the given table.

        Returns
        -------
        transformed_table : Table
            The transformed table.

        Raises
        ------
        TransformerNotFittedError
            If the transformer has not been fitted yet.

        Examples
        --------
        >>> from safeds.data.tabular.transformation import OneHotEncoder
        >>> from safeds.data.tabular.containers import Table
        >>> transformer = OneHotEncoder()
        >>> table = Table({"col1": [1, 2, 1], "col2": [1, 2, 4]})
        >>> fitted_transformer = transformer.fit(table, None)
        >>> table.transform_table(fitted_transformer)
           col1__1  col1__2  col2__1  col2__2  col2__4
        0      1.0      0.0      1.0      0.0      0.0
        1      0.0      1.0      0.0      1.0      0.0
        2      1.0      0.0      0.0      0.0      1.0
        """
        return transformer.transform(self)

    def inverse_transform_table(self, transformer: InvertibleTableTransformer) -> Table:
        """
        Invert the transformation applied by the given transformer.

        This table is not modified.

        Parameters
        ----------
        transformer : InvertibleTableTransformer
            A transformer that was fitted with columns, which are all present in the table.

        Returns
        -------
        table : Table
            The original table.

        Raises
        ------
        TransformerNotFittedError
            If the transformer has not been fitted yet.

        Examples
        --------
        >>> from safeds.data.tabular.transformation import OneHotEncoder
        >>> from safeds.data.tabular.containers import Table
        >>> transformer = OneHotEncoder()
        >>> table = Table({"col1": [1, 2, 1], "col2": [1, 2, 4]})
        >>> fitted_transformer = transformer.fit(table, None)
        >>> transformed_table = fitted_transformer.transform(table)
        >>> transformed_table.inverse_transform_table(fitted_transformer)
           col1  col2
        0     1     1
        1     2     2
        2     1     4
        >>> fitted_transformer.inverse_transform(transformed_table)
           col1  col2
        0     1     1
        1     2     2
        2     1     4
        """
        return transformer.inverse_transform(self)

    # ------------------------------------------------------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------------------------------------------------------

    def plot_correlation_heatmap(self) -> Image:
        """
        Plot a correlation heatmap for all numerical columns of this `Table`.

        Returns
        -------
        plot: Image
            The plot as an image.
        """
        only_numerical = self.remove_columns_with_non_numerical_values()

        fig = plt.figure()
        sns.heatmap(
            data=only_numerical._data.corr(),
            vmin=-1,
            vmax=1,
            xticklabels=only_numerical.column_names,
            yticklabels=only_numerical.column_names,
            cmap="vlag",
        )
        plt.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        plt.close()  # Prevents the figure from being displayed directly
        buffer.seek(0)
        return Image(buffer, format_=ImageFormat.PNG)

    def plot_lineplot(self, x_column_name: str, y_column_name: str) -> Image:
        """
        Plot two columns against each other in a lineplot.

        If there are multiple x-values for a y-value, the resulting plot will consist of a line representing the mean
        and the lower-transparency area around the line representing the 95% confidence interval.

        Parameters
        ----------
        x_column_name : str
            The column name of the column to be plotted on the x-Axis.
        y_column_name : str
            The column name of the column to be plotted on the y-Axis.

        Returns
        -------
        plot: Image
            The plot as an image.

        Raises
        ------
        UnknownColumnNameError
            If either of the columns do not exist.
        """
        if not self.has_column(x_column_name) or not self.has_column(y_column_name):
            raise UnknownColumnNameError(
                ([x_column_name] if not self.has_column(x_column_name) else [])
                + ([y_column_name] if not self.has_column(y_column_name) else []),
            )

        fig = plt.figure()
        ax = sns.lineplot(
            data=self._data,
            x=x_column_name,
            y=y_column_name,
        )
        ax.set(xlabel=x_column_name, ylabel=y_column_name)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment="right",
        )  # rotate the labels of the x Axis to prevent the chance of overlapping of the labels
        plt.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        plt.close()  # Prevents the figure from being displayed directly
        buffer.seek(0)
        return Image(buffer, format_=ImageFormat.PNG)

    def plot_scatterplot(self, x_column_name: str, y_column_name: str) -> Image:
        """
        Plot two columns against each other in a scatterplot.

        Parameters
        ----------
        x_column_name : str
            The column name of the column to be plotted on the x-Axis.
        y_column_name : str
            The column name of the column to be plotted on the y-Axis.

        Returns
        -------
        plot: Image
            The plot as an image.

        Raises
        ------
        UnknownColumnNameError
            If either of the columns do not exist.
        """
        if not self.has_column(x_column_name) or not self.has_column(y_column_name):
            raise UnknownColumnNameError(
                ([x_column_name] if not self.has_column(x_column_name) else [])
                + ([y_column_name] if not self.has_column(y_column_name) else []),
            )

        fig = plt.figure()
        ax = sns.scatterplot(
            data=self._data,
            x=x_column_name,
            y=y_column_name,
        )
        ax.set(xlabel=x_column_name, ylabel=y_column_name)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment="right",
        )  # rotate the labels of the x Axis to prevent the chance of overlapping of the labels
        plt.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        plt.close()  # Prevents the figure from being displayed directly
        buffer.seek(0)
        return Image(buffer, format_=ImageFormat.PNG)

    def plot_boxplots(self) -> Image:
        """
        Plot a boxplot for every numerical column.

        Returns
        -------
        plot: Image
            The plot as an image.

        Raises
        ------
        NonNumericColumnError
            If the table contains only non-numerical columns.
        """
        numerical_table = self.remove_columns_with_non_numerical_values()
        if numerical_table.number_of_columns == 0:
            raise NonNumericColumnError("This table contains only non-numerical columns.")
        col_wrap = min(numerical_table.number_of_columns, 3)

        data = pd.melt(numerical_table._data, value_vars=numerical_table.column_names)
        grid = sns.FacetGrid(data, col="variable", col_wrap=col_wrap, sharex=False, sharey=False)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Using the boxplot function without specifying `order` is likely to produce an incorrect plot.",
            )
            grid.map(sns.boxplot, "variable", "value")
        grid.set_xlabels("")
        grid.set_ylabels("")
        grid.set_titles("{col_name}")
        for axes in grid.axes.flat:
            axes.set_xticks([])
        plt.tight_layout()
        fig = grid.fig

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        plt.close()  # Prevents the figure from being displayed directly
        buffer.seek(0)
        return Image(buffer, format_=ImageFormat.PNG)

    def plot_histograms(self) -> Image:
        """
        Plot a histogram for every column.

        Returns
        -------
        plot: Image
            The plot as an image.
        """
        col_wrap = min(self.number_of_columns, 3)

        data = pd.melt(self._data, value_vars=self.column_names)
        grid = sns.FacetGrid(data=data, col="variable", col_wrap=col_wrap, sharex=False, sharey=False)
        grid.map(sns.histplot, "value")
        grid.set_xlabels("")
        grid.set_ylabels("")
        grid.set_titles("{col_name}")
        for axes in grid.axes.flat:
            axes.set_xticks(axes.get_xticks())
            axes.set_xticklabels(axes.get_xticklabels(), rotation=45, horizontalalignment="right")
        grid.tight_layout()
        fig = grid.fig

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)
        return Image(buffer, ImageFormat.PNG)

    # ------------------------------------------------------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------------------------------------------------------

    def to_csv_file(self, path: str | Path) -> None:
        """
        Write the data from the table into a CSV file.

        If the file and/or the directories do not exist they will be created. If the file already exists it will be
        overwritten.

        Parameters
        ----------
        path : str | Path
            The path to the output file.

        Raises
        ------
        WrongFileExtensionError
            If the file is not a csv file.
        """
        path = Path(path)
        if path.suffix != ".csv":
            raise WrongFileExtensionError(path, ".csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        data_to_csv = self._data.copy()
        data_to_csv.columns = self._schema.column_names
        data_to_csv.to_csv(path, index=False)

    def to_excel_file(self, path: str | Path) -> None:
        """
        Write the data from the table into an Excel file.

        Valid file extensions are `.xls`, '.xlsx', `.xlsm`, `.xlsb`, `.odf`, `.ods` and `.odt`.
        If the file and/or the directories do not exist, they will be created. If the file already exists, it will be
        overwritten.

        Parameters
        ----------
        path : str | Path
            The path to the output file.

        Raises
        ------
        WrongFileExtensionError
            If the file is not an Excel file.
        """
        path = Path(path)
        excel_extensions = [".xls", ".xlsx", ".xlsm", ".xlsb", ".odf", ".ods", ".odt"]
        if path.suffix not in excel_extensions:
            raise WrongFileExtensionError(path, excel_extensions)

        # Create Excel metadata in the file
        tmp_table_file = openpyxl.Workbook()
        tmp_table_file.save(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        data_to_excel = self._data.copy()
        data_to_excel.columns = self._schema.column_names
        data_to_excel.to_excel(path)

    def to_json_file(self, path: str | Path) -> None:
        """
        Write the data from the table into a JSON file.

        If the file and/or the directories do not exist, they will be created. If the file already exists it will be
        overwritten.

        Parameters
        ----------
        path : str | Path
            The path to the output file.

        Raises
        ------
        WrongFileExtensionError
            If the file is not a JSON file.
        """
        path = Path(path)
        if path.suffix != ".json":
            raise WrongFileExtensionError(path, ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        data_to_json = self._data.copy()
        data_to_json.columns = self._schema.column_names
        data_to_json.to_json(path)

    def to_dict(self) -> dict[str, list[Any]]:
        """
        Return a dictionary that maps column names to column values.

        Returns
        -------
        data : dict[str, list[Any]]
            Dictionary representation of the table.
        """
        return {column_name: list(self.get_column(column_name)) for column_name in self.column_names}

    def to_html(self) -> str:
        """
        Return an HTML representation of the table.

        Returns
        -------
        output : str
            The generated HTML.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> html = table.to_html()
        """
        return self._data.to_html(max_rows=self._data.shape[0], max_cols=self._data.shape[1])

    def to_columns(self) -> list[Column]:
        """
        Return a list of the columns.

        Returns
        -------
        columns : list[Columns]
            List of columns.
        """
        return [self.get_column(name) for name in self._schema.column_names]

    def to_rows(self) -> list[Row]:
        """
        Return a list of the rows.

        Returns
        -------
        rows : list[Row]
            List of rows.
        """
        return [
            Row._from_pandas_dataframe(
                pd.DataFrame([list(series_row)], columns=self._schema.column_names),
                self._schema,
            )
            for (_, series_row) in self._data.iterrows()
        ]

    # ------------------------------------------------------------------------------------------------------------------
    # IPython integration
    # ------------------------------------------------------------------------------------------------------------------

    def _repr_html_(self) -> str:
        """
        Return an HTML representation of the table.

        Returns
        -------
        output : str
            The generated HTML.
        """
        return self._data.to_html(max_rows=self._data.shape[0], max_cols=self._data.shape[1], notebook=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Dataframe interchange protocol
    # ------------------------------------------------------------------------------------------------------------------

    def __dataframe__(self, nan_as_null: bool = False, allow_copy: bool = True):  # type: ignore[no-untyped-def]
        """
        Return a DataFrame exchange object that conforms to the dataframe interchange protocol.

        Generally, there is no reason to call this method directly. The dataframe interchange protocol is designed to
        allow libraries to consume tabular data from different sources, such as `pandas` or `polars`. If you still
        decide to call this method, you should not rely on any capabilities of the returned object beyond the dataframe
        interchange protocol.

        The specification of the dataframe interchange protocol can be found on
        [GitHub](https://github.com/data-apis/dataframe-api).

        Parameters
        ----------
        nan_as_null : bool
            Whether to replace missing values in the data with `NaN`.
        allow_copy : bool
            Whether memory may be copied to create the DataFrame exchange object.

        Returns
        -------
        dataframe
            A DataFrame object that conforms to the dataframe interchange protocol.
        """
        if not allow_copy:
            raise NotImplementedError("For the moment we need to copy the data, so `allow_copy` must be True.")

        data_copy = self._data.copy()
        data_copy.columns = self.column_names
        return data_copy.__dataframe__(nan_as_null, allow_copy)
