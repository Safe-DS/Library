from __future__ import annotations

import functools
import os.path
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.core.display_functions import DisplayHandle, display
from pandas import DataFrame, Series
from safeds.data.tabular.typing import ColumnType, TableSchema
from safeds.exceptions import (
    ColumnLengthMismatchError,
    ColumnSizeError,
    DuplicateColumnNameError,
    IndexOutOfBoundsError,
    MissingDataError,
    MissingSchemaError,
    NonNumericColumnError,
    SchemaMismatchError,
    UnknownColumnNameError,
)
from scipy import stats

from ._column import Column
from ._row import Row

if TYPE_CHECKING:
    from ._tagged_table import TaggedTable


# noinspection PyProtectedMember
class Table:
    """
    A table is a two-dimensional collection of data. It can either be seen as a list of rows or as a list of columns.

    Parameters
    ----------
    data : typing.Iterable
        The data.
    schema : Optional[TableSchema]
        The schema of the table. If not specified, the schema will be inferred from the data.

    Raises
    ----------
    MissingSchemaError
        If the table is empty and no schema is specified.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def from_csv_file(path: str) -> Table:
        """
        Read data from a CSV file into a table.

        Parameters
        ----------
        path : str
            The path to the CSV file.

        Returns
        -------
        table : Table
            The table created from the CSV file.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        ValueError
            If the file could not be read.
        """

        try:
            return Table(pd.read_csv(path))
        except FileNotFoundError as exception:
            raise FileNotFoundError(f'File "{path}" does not exist') from exception
        except Exception as exception:
            raise ValueError(f'Could not read file from "{path}" as CSV') from exception

    @staticmethod
    def from_json_file(path: str) -> Table:
        """
        Read data from a JSON file into a table.

        Parameters
        ----------
        path : str
            The path to the JSON file.

        Returns
        -------
        table : Table
            The table created from the JSON file.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        ValueError
            If the file could not be read.
        """

        try:
            return Table(pd.read_json(path))
        except FileNotFoundError as exception:
            raise FileNotFoundError(f'File "{path}" does not exist') from exception
        except Exception as exception:
            raise ValueError(f'Could not read file from "{path}" as JSON') from exception

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
        MissingDataError
            If an empty list is given.
        ColumnLengthMismatchError
            If any of the column sizes does not match with the others.
        """
        if len(columns) == 0:
            raise MissingDataError("This function requires at least one column.")

        dataframe: DataFrame = pd.DataFrame()

        for column in columns:
            if column._data.size != columns[0]._data.size:
                raise ColumnLengthMismatchError(
                    "\n".join([f"{column.name}: {column._data.size}" for column in columns])
                )
            dataframe[column.name] = column._data

        return Table(dataframe)

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
        MissingDataError
            If an empty list is given.
        SchemaMismatchError
            If any of the row schemas does not match with the others.
        """
        if len(rows) == 0:
            raise MissingDataError("This function requires at least one row.")

        schema_compare: TableSchema = rows[0].schema
        row_array: list[Series] = []

        for row in rows:
            if schema_compare != row.schema:
                raise SchemaMismatchError()
            row_array.append(row._data)

        dataframe: DataFrame = pd.DataFrame(row_array)
        dataframe.columns = schema_compare.get_column_names()
        return Table(dataframe)

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, data: Iterable, schema: Optional[TableSchema] = None):
        self._data: pd.Dataframe = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        if schema is None:
            if self.count_columns() == 0:
                raise MissingSchemaError()
            self._schema: TableSchema = TableSchema._from_dataframe(self._data)
        else:
            self._schema = schema
            if self._data.empty:
                self._data = pd.DataFrame(columns=self._schema.get_column_names())

        self._data = self._data.reset_index(drop=True)
        self._data.columns = list(range(self.count_columns()))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Table):
            return NotImplemented
        if self is other:
            return True
        table1 = self.sort_columns()
        table2 = other.sort_columns()
        return table1._data.equals(table2._data) and table1._schema == table2._schema

    def __hash__(self) -> int:
        return hash(self._data)

    def __repr__(self) -> str:
        tmp = self._data.copy(deep=True)
        tmp.columns = self.get_column_names()
        return tmp.__repr__()

    def __str__(self) -> str:
        tmp = self._data.copy(deep=True)
        tmp.columns = self.get_column_names()
        return tmp.__str__()

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def schema(self) -> TableSchema:
        """
        Return the schema of the table.

        Returns
        -------
        schema : TableSchema
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
        if self._schema.has_column(column_name):
            output_column = Column(
                column_name,
                self._data.iloc[:, [self._schema._get_column_index_by_name(column_name)]].squeeze(),
                self._schema.get_type_of_column(column_name),
            )
            return output_column

        raise UnknownColumnNameError([column_name])

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

    def get_column_names(self) -> list[str]:
        """
        Return a list of all column names in this table.
        Alias for self.schema.get_column_names() -> list[str].

        Returns
        -------
        column_names : list[str]
            The list of the column names.
        """
        return self._schema.get_column_names()

    def get_type_of_column(self, column_name: str) -> ColumnType:
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
        ColumnNameError
            If the specified target column name does not exist.
        """
        return self._schema.get_type_of_column(column_name)

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
        return Row(self._data.iloc[[index]].squeeze(), self._schema)

    # ------------------------------------------------------------------------------------------------------------------
    # Information
    # ------------------------------------------------------------------------------------------------------------------

    def count_rows(self) -> int:
        """
        Return the number of rows.

        Returns
        -------
        count : int
            The number of rows.
        """
        return self._data.shape[0]

    def count_columns(self) -> int:
        """
        Return the number of columns.

        Returns
        -------
        count : int
            The number of columns.
        """
        return self._data.shape[1]

    def summary(self) -> Table:
        """
        Return a table with a number of statistical key values.

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
                "row count": column.count,
            }
            values = []

            for function in statistics.values():
                try:
                    values.append(str(function()))
                except NonNumericColumnError:
                    values.append("-")

            result = pd.concat([result, pd.DataFrame(values)], axis=1)

        result = pd.concat([pd.DataFrame(list(statistics.keys())), result], axis=1)
        result.columns = ["metrics"] + self.get_column_names()

        return Table(result)

    # ------------------------------------------------------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------------------------------------------------------

    def add_column(self, column: Column) -> Table:
        """
        Return the original table with the provided column attached at the end.

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
        if self._schema.has_column(column.name):
            raise DuplicateColumnNameError(column.name)

        if column._data.size != self.count_rows():
            raise ColumnSizeError(str(self.count_rows()), str(column._data.size))

        result = self._data.copy()
        result.columns = self._schema.get_column_names()
        result[column.name] = column._data
        return Table(result)

    def add_columns(self, columns: Union[list[Column], Table]) -> Table:
        """
        Add multiple columns to the table.

        Parameters
        ----------
        columns : list[Column] or Table
            The columns to be added.

        Returns
        -------
        result: Table
            A new table combining the original table and the given columns.

        Raises
        --------
        ColumnSizeError
            If at least one of the column sizes from the provided column list does not match the table.
        DuplicateColumnNameError
            If at least one column name from the provided column list already exists in the table.
        """
        if isinstance(columns, Table):
            columns = columns.to_columns()
        result = self._data.copy()
        result.columns = self._schema.get_column_names()
        for column in columns:
            if column.name in result.columns:
                raise DuplicateColumnNameError(column.name)

            if column._data.size != self.count_rows():
                raise ColumnSizeError(str(self.count_rows()), str(column._data.size))

            result[column.name] = column._data
        return Table(result)

    def add_row(self, row: Row) -> Table:
        """
        Add a row to the table.

        Parameters
        ----------
        row : Row
            The row to be added.

        Returns
        -------
        table : Table
            A new table with the added row at the end.

        """
        if self._schema != row.schema:
            raise SchemaMismatchError()
        new_df = pd.concat([self._data, row._data.to_frame().T]).infer_objects()
        new_df.columns = self._schema.get_column_names()
        return Table(new_df)

    def add_rows(self, rows: Union[list[Row], Table]) -> Table:
        """
        Add multiple rows to a table.

        Parameters
        ----------
        rows : list[Row] or Table
            The rows to be added.

        Returns
        -------
        result : Table
            A new table which combines the original table and the given rows.
        """
        if isinstance(rows, Table):
            rows = rows.to_rows()
        result = self._data
        for row in rows:
            if self._schema != row.schema:
                raise SchemaMismatchError()
        result = pd.concat([result, *[row._data.to_frame().T for row in rows]]).infer_objects()
        result.columns = self._schema.get_column_names()
        return Table(result)

    def drop_columns(self, column_names: list[str]) -> Table:
        """
        Return a table without the given column(s).

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
        ColumnNameError
            If any of the given columns do not exist.
        """
        invalid_columns = []
        column_indices = []
        for name in column_names:
            if not self._schema.has_column(name):
                invalid_columns.append(name)
            else:
                column_indices.append(self._schema._get_column_index_by_name(name))
        if len(invalid_columns) != 0:
            raise UnknownColumnNameError(invalid_columns)
        transformed_data = self._data.drop(labels=column_indices, axis="columns")
        transformed_data.columns = list(name for name in self._schema.get_column_names() if name not in column_names)
        return Table(transformed_data)

    def drop_columns_with_missing_values(self) -> Table:
        """
        Return a table without the columns that contain missing values.

        Returns
        -------
        table : Table
            A table without the columns that contain missing values.
        """
        return Table.from_columns([column for column in self.to_columns() if not column.has_missing_values()])

    def drop_columns_with_non_numerical_values(self) -> Table:
        """
        Return a table without the columns that contain non-numerical values.

        Returns
        -------
        table : Table
            A table without the columns that contain non-numerical values.

        """
        return Table.from_columns([column for column in self.to_columns() if column.type.is_numeric()])

    def drop_duplicate_rows(self) -> Table:
        """
        Return a copy of the table with every duplicate row removed.

        Returns
        -------
        result : Table
            The table with the duplicate rows removed.
        """
        df = self._data.drop_duplicates(ignore_index=True)
        df.columns = self._schema.get_column_names()
        return Table(df)

    def drop_rows_with_missing_values(self) -> Table:
        """
        Return a table without the rows that contain missing values.

        Returns
        -------
        table : Table
            A table without the rows that contain missing values.
        """
        result = self._data.copy(deep=True)
        result = result.dropna(axis="index")
        return Table(result, self._schema)

    def drop_rows_with_outliers(self) -> Table:
        """
        Remove all rows from the table that contain at least one outlier.

        We define an outlier as a value that has a distance of more than 3 standard deviations from the column mean.
        Missing values are not considered outliers. They are also ignored during the calculation of the standard
        deviation.

        Returns
        -------
        new_table : Table
            A new table without rows containing outliers.
        """
        copy = self._data.copy(deep=True)

        table_without_nonnumericals = self.drop_columns_with_non_numerical_values()
        z_scores = np.absolute(stats.zscore(table_without_nonnumericals._data, nan_policy="omit"))
        filter_ = ((z_scores < 3) | np.isnan(z_scores)).all(axis=1)

        return Table(copy[filter_], self._schema)

    def filter_rows(self, query: Callable[[Row], bool]) -> Table:
        """
        Return a table with rows filtered by Callable (e.g. lambda function).

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
            result_table = Table([], self._schema)
        else:
            result_table = self.from_rows(rows)
        return result_table

    def keep_only_columns(self, column_names: list[str]) -> Table:
        """
        Return a table with only the given column(s).

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
        ColumnNameError
            If any of the given columns do not exist.
        """
        invalid_columns = []
        column_indices = []
        for name in column_names:
            if not self._schema.has_column(name):
                invalid_columns.append(name)
            else:
                column_indices.append(self._schema._get_column_index_by_name(name))
        if len(invalid_columns) != 0:
            raise UnknownColumnNameError(invalid_columns)
        transformed_data = self._data[column_indices]
        transformed_data.columns = list(name for name in self._schema.get_column_names() if name in column_names)
        return Table(transformed_data)

    def rename_column(self, old_name: str, new_name: str) -> Table:
        """
        Rename a single column.

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
        ColumnNameError
            If the specified old target column name does not exist.
        DuplicateColumnNameError
            If the specified new target column name already exists.
        """
        if old_name not in self._schema.get_column_names():
            raise UnknownColumnNameError([old_name])
        if old_name == new_name:
            return self
        if new_name in self._schema.get_column_names():
            raise DuplicateColumnNameError(new_name)

        new_df = self._data.copy()
        new_df.columns = self._schema.get_column_names()
        return Table(new_df.rename(columns={old_name: new_name}))

    def replace_column(self, old_column_name: str, new_column: Column) -> Table:
        """
        Return a copy of the table with the specified old column replaced by a new column. Keeps the order of columns.

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
        if old_column_name not in self._schema.get_column_names():
            raise UnknownColumnNameError([old_column_name])

        if new_column.name in self._schema.get_column_names() and new_column.name != old_column_name:
            raise DuplicateColumnNameError(new_column.name)

        if self.count_rows() != new_column._data.size:
            raise ColumnSizeError(str(self.count_rows()), str(new_column._data.size))

        if old_column_name != new_column.name:
            renamed_table = self.rename_column(old_column_name, new_column.name)
            result = renamed_table._data
            result.columns = renamed_table._schema.get_column_names()
        else:
            result = self._data.copy()
            result.columns = self._schema.get_column_names()

        result[new_column.name] = new_column._data
        return Table(result)

    def shuffle(self) -> Table:
        """
        Shuffle the table randomly.

        Returns
        -------
        result : Table
            The shuffled Table.

        """
        new_df = self._data.sample(frac=1.0)
        new_df.columns = self._schema.get_column_names()
        return Table(new_df)

    def slice(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        step: int = 1,
    ) -> Table:
        """
        Slice a part of the table into a new table.

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
        -------
        ValueError
            If the index is out of bounds.
        """

        if start is None:
            start = 0

        if end is None:
            end = self.count_rows()

        if start < 0 or end < 0 or start >= self.count_rows() or end > self.count_rows() or end < start:
            raise ValueError("the given index is out of bounds")

        new_df = self._data.iloc[start:end:step]
        new_df.columns = self._schema.get_column_names()
        return Table(new_df)

    def sort_columns(
        self,
        comparator: Callable[[Column, Column], int] = lambda col1, col2: (col1.name > col2.name)
        - (col1.name < col2.name),
    ) -> Table:
        """
        Sort the columns of a `Table` with the given comparator and return a new `Table`. The original table is not
        modified.

        The comparator is a function that takes two columns `col1` and `col2` and returns an integer:

        * If `col1` should be ordered before `col2`, the function should return a negative number.
        * If `col1` should be ordered after `col2`, the function should return a positive number.
        * If the original order of `col1` and `col2` should be kept, the function should return 0.

        If no comparator is given, the columns will be sorted alphabetically by their name.

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
        Sort the rows of a `Table` with the given comparator and return a new `Table`. The original table is not
        modified.

        The comparator is a function that takes two rows `row1` and `row2` and returns an integer:

        * If `row1` should be ordered before `row2`, the function should return a negative number.
        * If `row1` should be ordered after `row2`, the function should return a positive number.
        * If the original order of `row1` and `row2` should be kept, the function should return 0.

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

        Parameters
        -------
        percentage_in_first : float
            The desired size of the first table in percentage to the given table.

        Returns
        -------
        result : (Table, Table)
            A tuple containing the two resulting tables. The first table has the specified size, the second table
            contains the rest of the data.


        """
        if percentage_in_first <= 0 or percentage_in_first >= 1:
            raise ValueError("the given percentage is not in range")
        return (
            self.slice(0, round(percentage_in_first * self.count_rows())),
            self.slice(round(percentage_in_first * self.count_rows())),
        )

    def tag_columns(self, target_name: str, feature_names: Optional[list[str]] = None) -> TaggedTable:
        """
        Mark the columns of the table as target column or feature columns. The original table is not modified.

        Parameters
        ----------
        target_name : str
            Name of the target column.
        feature_names : Optional[list[str]]
            Names of the feature columns. If None, all columns except the target column are used.

        Returns
        -------
        tagged_table : TaggedTable
            A new tagged table with the given target and feature names.
        """
        # pylint: disable=import-outside-toplevel
        from ._tagged_table import TaggedTable

        return TaggedTable(self._data, target_name, feature_names, self._schema)

    def transform_column(self, name: str, transformer: Callable[[Row], Any]) -> Table:
        """
        Transform provided column by calling provided transformer.

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
            result: Column = Column(name, pd.Series(items))
            return self.replace_column(name, result)
        raise UnknownColumnNameError([name])

    # ------------------------------------------------------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------------------------------------------------------

    def correlation_heatmap(self) -> None:
        """
        Plot a correlation heatmap for all numerical columns of this `Table`.
        """
        only_numerical = self.drop_columns_with_non_numerical_values()

        sns.heatmap(
            data=only_numerical._data.corr(),
            vmin=-1,
            vmax=1,
            xticklabels=only_numerical.get_column_names(),
            yticklabels=only_numerical.get_column_names(),
            cmap="vlag",
        )
        plt.tight_layout()
        plt.show()

    def lineplot(self, x_column_name: str, y_column_name: str) -> None:
        """
        Plot two columns against each other in a lineplot. If there are multiple x-values for a y-value,
        the resulting plot will consist of a line representing the mean and the lower-transparency area around the line
        representing the 95% confidence interval.

        Parameters
        ----------
        x_column_name : str
            The column name of the column to be plotted on the x-Axis.
        y_column_name : str
            The column name of the column to be plotted on the y-Axis.

        Raises
        ---------
        UnknownColumnNameError
            If either of the columns do not exist.
        """
        if not self.has_column(x_column_name):
            raise UnknownColumnNameError([x_column_name])
        if not self.has_column(y_column_name):
            raise UnknownColumnNameError([y_column_name])

        ax = sns.lineplot(
            data=self._data,
            x=self._schema._get_column_index_by_name(x_column_name),
            y=self._schema._get_column_index_by_name(y_column_name),
        )
        ax.set(xlabel=x_column_name, ylabel=y_column_name)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, horizontalalignment="right"
        )  # rotate the labels of the x Axis to prevent the chance of overlapping of the labels
        plt.tight_layout()
        plt.show()

    def scatterplot(self, x_column_name: str, y_column_name: str) -> None:
        """
        Plot two columns against each other in a scatterplot.

        Parameters
        ----------
        x_column_name : str
            The column name of the column to be plotted on the x-Axis.
        y_column_name : str
            The column name of the column to be plotted on the y-Axis.

        Raises
        ---------
        UnknownColumnNameError
            If either of the columns do not exist.
        """
        if not self.has_column(x_column_name):
            raise UnknownColumnNameError([x_column_name])
        if not self.has_column(y_column_name):
            raise UnknownColumnNameError([y_column_name])

        ax = sns.scatterplot(
            data=self._data,
            x=self._schema._get_column_index_by_name(x_column_name),
            y=self._schema._get_column_index_by_name(y_column_name),
        )
        ax.set(xlabel=x_column_name, ylabel=y_column_name)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, horizontalalignment="right"
        )  # rotate the labels of the x Axis to prevent the chance of overlapping of the labels
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------------------------------------------------------

    def to_csv_file(self, path: str) -> None:
        """
        Write the data from the table into a CSV file.
        If the file and/or the directories do not exist they will be created.
        If the file already exists it will be overwritten.

        Parameters
        ----------
        path : str
            The path to the output file.
        """
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        data_to_csv = self._data.copy()
        data_to_csv.columns = self._schema.get_column_names()
        data_to_csv.to_csv(path, index=False)

    def to_json_file(self, path: str) -> None:
        """
        Write the data from the table into a JSON file.
        If the file and/or the directories do not exist, they will be created.
        If the file already exists it will be overwritten.

        Parameters
        ----------
        path : str
            The path to the output file.
        """
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        data_to_json = self._data.copy()
        data_to_json.columns = self._schema.get_column_names()
        data_to_json.to_json(path)

    def to_columns(self) -> list[Column]:
        """
        Return a list of the columns.

        Returns
        -------
        columns : list[Columns]
            List of columns.
        """
        return [self.get_column(name) for name in self._schema.get_column_names()]

    def to_rows(self) -> list[Row]:
        """
        Return a list of the rows.

        Returns
        -------
        rows : list[Row]
            List of rows.
        """
        return [Row(series_row, self._schema) for (_, series_row) in self._data.iterrows()]

    # ------------------------------------------------------------------------------------------------------------------
    # Other
    # ------------------------------------------------------------------------------------------------------------------

    def _ipython_display_(self) -> DisplayHandle:
        """
        Return a display object for the column to be used in Jupyter Notebooks.

        Returns
        -------
        output : DisplayHandle
            Output object.
        """
        tmp = self._data.copy(deep=True)
        tmp.columns = self.get_column_names()

        with pd.option_context("display.max_rows", tmp.shape[0], "display.max_columns", tmp.shape[1]):
            return display(tmp)
