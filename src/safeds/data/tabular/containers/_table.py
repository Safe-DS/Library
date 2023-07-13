from __future__ import annotations

import copy
import functools
import io
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import Levenshtein
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

    Note: When removing the last column of the table, the `number_of_columns` property will be set to 0.

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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> Table.from_csv_file('./src/resources/from_csv_file.csv')
           a  b  c
        0  1  2  1
        1  0  0  7
        """
        path = Path(path)
        if path.suffix != ".csv":
            raise WrongFileExtensionError(path, ".csv")
        if path.exists():
            with path.open() as f:
                if f.read().replace("\n", "") == "":
                    return Table()

            return Table._from_pandas_dataframe(pd.read_csv(path))
        else:
            raise FileNotFoundError(f'File "{path}" does not exist')

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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> Table.from_excel_file('./src/resources/from_excel_file.xlsx')
           a  b
        0  1  4
        1  2  5
        2  3  6
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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> Table.from_json_file('./src/resources/from_json_file.json')
           a  b
        0  1  4
        1  2  5
        2  3  6
        """
        path = Path(path)
        if path.suffix != ".json":
            raise WrongFileExtensionError(path, ".json")
        if path.exists():
            with path.open() as f:
                if f.read().replace("\n", "") in ("", "{}"):
                    return Table()

            return Table._from_pandas_dataframe(pd.read_json(path))
        else:
            raise FileNotFoundError(f'File "{path}" does not exist')

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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> d = {'a': [1, 2], 'b': [3, 4]}
        >>> Table.from_dict(d)
           a  b
        0  1  3
        1  2  4
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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column, Table
        >>> col1 = Column("a", [1, 2, 3])
        >>> col2 = Column("b", [4, 5, 6])
        >>> Table.from_columns([col1, col2])
           a  b
        0  1  4
        1  2  5
        2  3  6
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
        UnknownColumnNameError
            If any of the row column names does not match with the first row.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row, Table
        >>> row1 = Row({"a": 1, "b": 2})
        >>> row2 = Row({"a": 3, "b": 4})
        >>> Table.from_rows([row1, row2])
           a  b
        0  1  2
        1  3  4
        """
        if len(rows) == 0:
            return Table._from_pandas_dataframe(pd.DataFrame())

        column_names_compare: list = list(rows[0].column_names)
        unknown_column_names = set()
        row_array: list[pd.DataFrame] = []

        for row in rows:
            unknown_column_names.update(set(column_names_compare) - set(row.column_names))
            row_array.append(row._data)
        if len(unknown_column_names) > 0:
            raise UnknownColumnNameError(list(unknown_column_names))

        dataframe: DataFrame = pd.concat(row_array, ignore_index=True)
        dataframe.columns = column_names_compare

        schema = Schema.merge_multiple_schemas([row.schema for row in rows])

        return Table._from_pandas_dataframe(dataframe, schema)

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
        >>> Table._from_pandas_dataframe(pd.DataFrame({"a": [1], "b": [2]}))
           a  b
        0  1  2
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
        >>> Table({"a": [1, 2, 3], "b": [4, 5, 6]})
           a  b
        0  1  4
        1  2  5
        2  3  6
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
        """
        Compare two table instances.

        Returns
        -------
        'True' if contents are equal, 'False' otherwise.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row, Table
        >>> row1 = Row({"a": 1, "b": 2})
        >>> row2 = Row({"a": 3, "b": 4})
        >>> row3 = Row({"a": 5, "b": 6})
        >>> table1 = Table.from_rows([row1, row2])
        >>> table2 = Table.from_rows([row1, row2])
        >>> table3 = Table.from_rows([row1, row3])
        >>> table1 == table2
        True
        >>> table1 == table3
        False
        """
        if not isinstance(other, Table):
            return NotImplemented
        if self is other:
            return True
        if self.number_of_columns == 0 and other.number_of_columns == 0:
            return True
        table1 = self.sort_columns()
        table2 = other.sort_columns()
        if table1.number_of_rows == 0 and table2.number_of_rows == 0:
            return table1.column_names == table2.column_names
        return table1._schema == table2._schema and table1._data.equals(table2._data)

    def __repr__(self) -> str:
        r"""
        Display the table in only one line.

        Returns
        -------
        A string representation of the table in only one line.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"a": [1, 3], "b": [2, 4]})
        >>> repr(table)
        '   a  b\n0  1  2\n1  3  4'
        """
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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"col1": [1, 3], "col2": [2, 4]})
        >>> table.column_names
        ['col1', 'col2']
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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"a": [1], "b": [2]})
        >>> table.number_of_columns
        2
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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"a": [1], "b": [2]})
        >>> table.number_of_rows
        1
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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row, Table
        >>> row = Row({"a": 1, "b": 2.5, "c": "ff"})
        >>> table = Table.from_dict({"a": [1, 8], "b": [2.5, 9], "c": ["g", "g"]})
        >>> table.schema
        Schema({
            'a': Integer,
            'b': RealNumber,
            'c': String
        })
        >>> table.schema == row.schema
        True
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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"a": [1], "b": [2]})
        >>> table.get_column("b")
        Column('b', [2])
        """
        if not self.has_column(column_name):
            similar_columns = self._get_similar_columns(column_name)
            raise UnknownColumnNameError([column_name], similar_columns)

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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"a": [1], "b": [2]})
        >>> table.has_column("b")
        True
        >>> table.has_column("c")
        False
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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"a": [1], "b": [2.5]})
        >>> table.get_column_type("b")
        RealNumber
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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"a": [1, 3], "b": [2, 4]})
        >>> table.get_row(0)
        Row({
            'a': 1,
            'b': 2
        })
        """
        if len(self._data.index) - 1 < index or index < 0:
            raise IndexOutOfBoundsError(index)

        return Row._from_pandas_dataframe(self._data.iloc[[index]], self._schema)

    def _get_similar_columns(self, column_name: str) -> list[str]:
        """
        Get all the column names in a Table that are similar to a given name.

        Parameters
        ----------
        column_name : str
            The name to compare the Table's column names to.

        Returns
        -------
        similar_columns: list[str]
            A list of all column names in the Table that are similar or equal to the given column name.
        """
        similar_columns = []
        similarity = 0.6
        i = 0
        while i < len(self.column_names):
            if Levenshtein.jaro_winkler(self.column_names[i], column_name) >= similarity:
                similar_columns.append(self.column_names[i])
            i += 1
            if len(similar_columns) == 4 and similarity < 0.9:
                similarity += 0.1
                similar_columns = []
                i = 0

        return similar_columns

    # ------------------------------------------------------------------------------------------------------------------
    # Information
    # ------------------------------------------------------------------------------------------------------------------

    def summarize_statistics(self) -> Table:
        """
        Return a table with a number of statistical key values.

        The original table is not modified.

        Returns
        -------
        result : Table
            The table with statistics.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"a": [1, 3], "b": [2, 4]})
        >>> table.summarize_statistics()
                      metrics                   a                   b
        0             maximum                   3                   4
        1             minimum                   1                   2
        2                mean                 2.0                 3.0
        3                mode              [1, 3]              [2, 4]
        4              median                 2.0                 3.0
        5                 sum                   4                   6
        6            variance                 2.0                 2.0
        7  standard deviation  1.4142135623730951  1.4142135623730951
        8              idness                 1.0                 1.0
        9           stability                 0.5                 0.5
        """
        if self.number_of_columns == 0:
            return Table(
                {
                    "metrics": [
                        "maximum",
                        "minimum",
                        "mean",
                        "mode",
                        "median",
                        "sum",
                        "variance",
                        "standard deviation",
                        "idness",
                        "stability",
                    ],
                },
            )
        elif self.number_of_rows == 0:
            table = Table(
                {
                    "metrics": [
                        "maximum",
                        "minimum",
                        "mean",
                        "mode",
                        "median",
                        "sum",
                        "variance",
                        "standard deviation",
                        "idness",
                        "stability",
                    ],
                },
            )
            for name in self.column_names:
                table = table.add_column(Column(name, ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]))
            return table

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
                except (NonNumericColumnError, ValueError):
                    values.append("-")

            result = pd.concat([result, pd.DataFrame(values)], axis=1)

        result = pd.concat([pd.DataFrame(list(statistics.keys())), result], axis=1)
        result.columns = ["metrics", *self.column_names]

        return Table._from_pandas_dataframe(result)

    # ------------------------------------------------------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------------------------------------------------------

    # This method is meant as a way to "cast" instances of subclasses of `Table` to a proper `Table`, dropping any
    # additional constraints that might have to hold in the subclass.
    # Override accordingly in subclasses.
    def _as_table(self: Table) -> Table:
        """
        Transform the table to an instance of the Table class.

        Returns
        -------
        table: Table
            The table, as an instance of the Table class.
        """
        return self

    def add_column(self, column: Column) -> Table:
        """
        Return a new table with the provided column attached at the end.

        The original table is not modified.

        Returns
        -------
        result : Table
            The table with the column attached.

        Raises
        ------
        DuplicateColumnNameError
            If the new column already exists.
        ColumnSizeError
            If the size of the column does not match the number of rows.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"a": [1, 3], "b": [2, 4]})
        >>> col = Column("c", ["d", "e"])
        >>> table.add_column(col)
           a  b  c
        0  1  2  d
        1  3  4  e
        """
        if self.has_column(column.name):
            raise DuplicateColumnNameError(column.name)

        if column.number_of_rows != self.number_of_rows and self.number_of_columns != 0:
            raise ColumnSizeError(str(self.number_of_rows), str(column._data.size))

        result = self._data.copy()
        result.columns = self._schema.column_names
        result[column.name] = column._data
        return Table._from_pandas_dataframe(result)

    def add_columns(self, columns: list[Column] | Table) -> Table:
        """
        Return a new `Table` with multiple added columns.

        The original table is not modified.

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
        DuplicateColumnNameError
            If at least one column name from the provided column list already exists in the table.
        ColumnSizeError
            If at least one of the column sizes from the provided column list does not match the table.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column, Table
        >>> table = Table.from_dict({"a": [1, 3], "b": [2, 4]})
        >>> col1 = Column("c", ["d", "e"])
        >>> col2 = Column("d", [3.5, 7.9])
        >>> table.add_columns([col1, col2])
           a  b  c    d
        0  1  2  d  3.5
        1  3  4  e  7.9
        """
        if isinstance(columns, Table):
            columns = columns.to_columns()
        result = self._data.copy()
        result.columns = self._schema.column_names
        for column in columns:
            if column.name in result.columns:
                raise DuplicateColumnNameError(column.name)

            if column.number_of_rows != self.number_of_rows and self.number_of_columns != 0:
                raise ColumnSizeError(str(self.number_of_rows), str(column._data.size))

            result[column.name] = column._data
        return Table._from_pandas_dataframe(result)

    def add_row(self, row: Row) -> Table:
        """
        Return a new `Table` with an added Row attached.

        If the table happens to be empty beforehand, respective columns will be added automatically.

        The order of columns of the new row will be adjusted to the order of columns in the table.
        The new table will contain the merged schema.

        The original table is not modified.

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
        UnknownColumnNameError
            If the row has different column names than the table.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row, Table
        >>> table = Table.from_dict({"a": [1], "b": [2]})
        >>> row = Row.from_dict({"a": 3, "b": 4})
        >>> table.add_row(row)
           a  b
        0  1  2
        1  3  4
        """
        int_columns = []
        result = self._copy()
        if self.number_of_columns == 0:
            return Table.from_rows([row])
        if len(set(self.column_names) - set(row.column_names)) > 0:
            raise UnknownColumnNameError(
                sorted(
                    set(self.column_names) - set(row.column_names),
                    key={val: ix for ix, val in enumerate(self.column_names)}.__getitem__,
                ),
            )

        if result.number_of_rows == 0:
            int_columns = list(filter(lambda name: isinstance(row[name], int | np.int64 | np.int32), row.column_names))

        new_df = pd.concat([result._data, row._data]).infer_objects()
        new_df.columns = result.column_names
        schema = Schema.merge_multiple_schemas([result.schema, row.schema])
        result = Table._from_pandas_dataframe(new_df, schema)

        for column in int_columns:
            result = result.replace_column(column, [result.get_column(column).transform(lambda it: int(it))])

        return result

    def add_rows(self, rows: list[Row] | Table) -> Table:
        """
        Return a new `Table` with multiple added Rows attached.

        The order of columns of the new rows will be adjusted to the order of columns in the table.
        The new table will contain the merged schema.

        The original table is not modified.

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
        UnknownColumnNameError
            If at least one of the rows have different column names than the table.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Row, Table
        >>> table = Table.from_dict({"a": [1], "b": [2]})
        >>> row1 = Row.from_dict({"a": 3, "b": 4})
        >>> row2 = Row.from_dict({"a": 5, "b": 6})
        >>> table.add_rows([row1, row2])
           a  b
        0  1  2
        1  3  4
        2  5  6
        """
        if isinstance(rows, Table):
            rows = rows.to_rows()

        if len(rows) == 0:
            return self._copy()

        different_column_names = set()
        for row in rows:
            different_column_names.update(set(self.column_names) - set(row.column_names))
        if len(different_column_names) > 0:
            raise UnknownColumnNameError(
                sorted(
                    different_column_names,
                    key={val: ix for ix, val in enumerate(self.column_names)}.__getitem__,
                ),
            )

        result = self._copy()

        for row in rows:
            result = result.add_row(row)

        return result

    def filter_rows(self, query: Callable[[Row], bool]) -> Table:
        """
        Return a new table with rows filtered by Callable (e.g. lambda function).

        The original table is not modified.

        Parameters
        ----------
        query : lambda function
            A Callable that is applied to all rows.

        Returns
        -------
        table : Table
            A table containing only the rows filtered by the query.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"a": [1, 3], "b": [2, 4]})
        >>> table.filter_rows(lambda x: x["a"] < 2)
           a  b
        0  1  2
        """
        rows: list[Row] = [row for row in self.to_rows() if query(row)]
        if len(rows) == 0:
            result_table = Table._from_pandas_dataframe(pd.DataFrame(), self._schema)
        else:
            result_table = self.from_rows(rows)
        return result_table

    _T = TypeVar("_T")

    def group_rows_by(self, key_selector: Callable[[Row], _T]) -> dict[_T, Table]:
        """
        Return a dictionary with copies of the output tables as values and the keys from the key_selector.

        The original table is not modified.

        Parameters
        ----------
        key_selector : Callable[[Row], _T]
            A Callable that is applied to all rows and returns the key of the group.

        Returns
        -------
        dictionary : dict
            A dictionary containing the new tables as values and the selected keys as keys.
        """
        dictionary: dict[Table._T, Table] = {}
        for row in self.to_rows():
            if key_selector(row) in dictionary:
                dictionary[key_selector(row)] = dictionary[key_selector(row)].add_row(row)
            else:
                dictionary[key_selector(row)] = Table.from_rows([row])
        return dictionary

    def keep_only_columns(self, column_names: list[str]) -> Table:
        """
        Return a new table with only the given column(s).

        The original table is not modified.

        Note: When removing the last column of the table, the `number_of_columns` property will be set to 0.

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
        IllegalSchemaModificationError
            If removing the columns would violate an invariant in the subclass.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"a": [1, 3], "b": [2, 4]})
        >>> table.keep_only_columns(["b"])
           b
        0  2
        1  4
        """
        invalid_columns = []
        similar_columns: list[str] = []
        for name in column_names:
            if not self._schema.has_column(name):
                similar_columns = similar_columns + self._get_similar_columns(name)
                invalid_columns.append(name)
        if len(invalid_columns) != 0:
            raise UnknownColumnNameError(invalid_columns, similar_columns)

        clone = self._copy()
        clone = clone.remove_columns(list(set(self.column_names) - set(column_names)))
        return clone

    def remove_columns(self, column_names: list[str]) -> Table:
        """
        Return a new table without the given column(s).

        The original table is not modified.

        Note: When removing the last column of the table, the `number_of_columns` property will be set to 0.

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
        IllegalSchemaModificationError
            If removing the columns would violate an invariant in the subclass.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"a": [1, 3], "b": [2, 4]})
        >>> table.remove_columns(["b"])
           a
        0  1
        1  3
        """
        invalid_columns = []
        similar_columns: list[str] = []
        for name in column_names:
            if not self._schema.has_column(name):
                similar_columns = similar_columns + self._get_similar_columns(name)
                invalid_columns.append(name)
        if len(invalid_columns) != 0:
            raise UnknownColumnNameError(invalid_columns, similar_columns)

        transformed_data = self._data.drop(labels=column_names, axis="columns")
        transformed_data.columns = [name for name in self._schema.column_names if name not in column_names]

        if len(transformed_data.columns) == 0:
            return Table()

        return Table._from_pandas_dataframe(transformed_data)

    def remove_columns_with_missing_values(self) -> Table:
        """
        Return a new table without the columns that contain missing values.

        The original table is not modified.

        Note: When removing the last column of the table, the `number_of_columns` property will be set to 0.

        Returns
        -------
        table : Table
            A table without the columns that contain missing values.

        Raises
        ------
        IllegalSchemaModificationError
            If removing the columns would violate an invariant in the subclass.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"a": [1, 2], "b": [None, 2]})
        >>> table.remove_columns_with_missing_values()
           a
        0  1
        1  2
        """
        return Table.from_columns([column for column in self.to_columns() if not column.has_missing_values()])

    def remove_columns_with_non_numerical_values(self) -> Table:
        """
        Return a new table without the columns that contain non-numerical values.

        The original table is not modified.

        Note: When removing the last column of the table, the `number_of_columns` property will be set to 0.

        Returns
        -------
        table : Table
            A table without the columns that contain non-numerical values.

        Raises
        ------
        IllegalSchemaModificationError
            If removing the columns would violate an invariant in the subclass.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"a": [1, 0], "b": ["test", 2]})
        >>> table.remove_columns_with_non_numerical_values()
           a
        0  1
        1  0
        """
        return Table.from_columns([column for column in self.to_columns() if column.type.is_numeric()])

    def remove_duplicate_rows(self) -> Table:
        """
        Return a new table with every duplicate row removed.

        The original table is not modified.

        Returns
        -------
        result : Table
            The table with the duplicate rows removed.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"a": [1, 3, 3], "b": [2, 4, 4]})
        >>> table.remove_duplicate_rows()
           a  b
        0  1  2
        1  3  4
        """
        result = self._data.drop_duplicates(ignore_index=True)
        result.columns = self._schema.column_names
        return Table._from_pandas_dataframe(result)

    def remove_rows_with_missing_values(self) -> Table:
        """
        Return a new table without the rows that contain missing values.

        The original table is not modified.

        Returns
        -------
        table : Table
            A table without the rows that contain missing values.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"a": [1.0, None, 3], "b": [2, 4.0, None]})
        >>> table.remove_rows_with_missing_values()
             a    b
        0  1.0  2.0
        """
        result = self._data.copy(deep=True)
        result = result.dropna(axis="index")
        return Table._from_pandas_dataframe(result)

    def remove_rows_with_outliers(self) -> Table:
        """
        Return a new table without those rows that contain at least one outlier.

        We define an outlier as a value that has a distance of more than 3 standard deviations from the column mean.
        Missing values are not considered outliers. They are also ignored during the calculation of the standard
        deviation.

        The original table is not modified.

        Returns
        -------
        new_table : Table
            A new table without rows containing outliers.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column, Table
        >>> c1 = Column("a", [1, 3, 1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0])
        >>> c2 = Column("b", [1.5, 1, 0.5, 0.01, 0, 0, 0, 0, 0, 0, 0, 0])
        >>> c3 = Column("c", [0.1, 0.00, 0.4, 0.2, 0, 0, 0, 0, 0, 0, 0, 0])
        >>> c4 = Column("d", [-1000000, 1000000, -1000000, -1000000, -1000000, -1000000, -1000000, -1000000, -1000000, -1000000, -1000000, -1000000])
        >>> table = Table.from_columns([c1, c2, c3, c4])
        >>> table.remove_rows_with_outliers()
              a     b    c        d
        0   1.0  1.50  0.1 -1000000
        1   1.0  0.50  0.4 -1000000
        2   0.1  0.01  0.2 -1000000
        3   0.0  0.00  0.0 -1000000
        4   0.0  0.00  0.0 -1000000
        5   0.0  0.00  0.0 -1000000
        6   0.0  0.00  0.0 -1000000
        7   0.0  0.00  0.0 -1000000
        8   0.0  0.00  0.0 -1000000
        9   0.0  0.00  0.0 -1000000
        10  0.0  0.00  0.0 -1000000
        """
        copy = self._data.copy(deep=True)

        table_without_nonnumericals = self.remove_columns_with_non_numerical_values()
        z_scores = np.absolute(stats.zscore(table_without_nonnumericals._data, nan_policy="omit"))
        filter_ = ((z_scores < 3) | np.isnan(z_scores)).all(axis=1)

        return Table._from_pandas_dataframe(copy[filter_], self._schema)

    def rename_column(self, old_name: str, new_name: str) -> Table:
        """
        Return a new `Table` with a single column renamed.

        The original table is not modified.

        Parameters
        ----------
        old_name : str
            The old name of the target column.
        new_name : str
            The new name of the target column.

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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"a": [1], "b": [2]})
        >>> table.rename_column("b", "c")
           a  c
        0  1  2
        """
        if old_name not in self._schema.column_names:
            similar_columns = self._get_similar_columns(old_name)
            raise UnknownColumnNameError([old_name], similar_columns)
        if old_name == new_name:
            return self
        if new_name in self._schema.column_names:
            raise DuplicateColumnNameError(new_name)

        new_df = self._data.copy()
        new_df.columns = self._schema.column_names
        return Table._from_pandas_dataframe(new_df.rename(columns={old_name: new_name}))

    def replace_column(self, old_column_name: str, new_columns: list[Column]) -> Table:
        """
        Return a new table with the specified old column replaced by a list of new columns.

        The order of columns is kept.

        The original table is not modified.

        Parameters
        ----------
        old_column_name : str
            The name of the column to be replaced.

        new_columns : list[Column]
            The list of new columns replacing the old column.

        Returns
        -------
        result : Table
            A table with the old column replaced by the new columns.

        Raises
        ------
        UnknownColumnNameError
            If the old column does not exist.
        DuplicateColumnNameError
            If at least one of the new columns already exists and the existing column is not affected by the replacement.
        ColumnSizeError
            If the size of at least one of the new columns does not match the amount of rows.
        IllegalSchemaModificationError
            If replacing the column would violate an invariant in the subclass.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column, Table
        >>> table = Table.from_dict({"a": [1], "b": [2]})
        >>> new_col = Column("new", [3])
        >>> table.replace_column("b", [new_col])
           a  new
        0  1    3
        """
        if old_column_name not in self._schema.column_names:
            similar_columns = self._get_similar_columns(old_column_name)
            raise UnknownColumnNameError([old_column_name], similar_columns)

        columns = list[Column]()
        for old_column in self.column_names:
            if old_column == old_column_name:
                for new_column in new_columns:
                    if new_column.name in self.column_names and new_column.name != old_column_name:
                        raise DuplicateColumnNameError(new_column.name)

                    if self.number_of_rows != new_column.number_of_rows:
                        raise ColumnSizeError(str(self.number_of_rows), str(new_column.number_of_rows))
                    columns.append(new_column)
            else:
                columns.append(self.get_column(old_column))

        return Table.from_columns(columns)

    def shuffle_rows(self) -> Table:
        """
        Return a new `Table` with randomly shuffled rows of this `Table`.

        The original table is not modified.

        Returns
        -------
        result : Table
            The shuffled Table.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> import numpy as np
        >>> np.random.seed(123456)
        >>> table = Table.from_dict({"a": [1, 3, 5], "b": [2, 4, 6]})
        >>> table.shuffle_rows()
           a  b
        0  5  6
        1  1  2
        2  3  4
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

        The original table is not modified.

        Parameters
        ----------
        start : int | None
            The first index of the range to be copied into a new table, None by default.
        end : int | None
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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"a": [1, 3, 5], "b": [2, 4, 6]})
        >>> table.slice_rows(0, 2)
           a  b
        0  1  2
        1  3  4
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

        The comparator is a function that takes two columns `col1` and `col2` and
        returns an integer:

        * If `col1` should be ordered before `col2`, the function should return a negative number.
        * If `col1` should be ordered after `col2`, the function should return a positive number.
        * If the original order of `col1` and `col2` should be kept, the function should return 0.

        If no comparator is given, the columns will be sorted alphabetically by their name.

        The original table is not modified.

        Parameters
        ----------
        comparator : Callable[[Column, Column], int]
            The function used to compare two columns.

        Returns
        -------
        new_table : Table
            A new table with sorted columns.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"a": [1], "b": [2] })
        >>> table.sort_columns(lambda col1, col2: 1)
           a  b
        0  1  2
        >>> table.sort_columns(lambda col1, col2: -1)
           b  a
        0  2  1
        >>> table2 = Table.from_dict({"b": [2], "a": [1]})
        >>> table2.sort_columns()
           a  b
        0  1  2
        """
        columns = self.to_columns()
        columns.sort(key=functools.cmp_to_key(comparator))
        return Table.from_columns(columns)

    def sort_rows(self, comparator: Callable[[Row, Row], int]) -> Table:
        """
        Sort the rows of a `Table` with the given comparator and return a new `Table`.

        The comparator is a function that takes two rows `row1` and `row2` and
        returns an integer:

        * If `row1` should be ordered before `row2`, the function should return a negative number.
        * If `row1` should be ordered after `row2`, the function should return a positive number.
        * If the original order of `row1` and `row2` should be kept, the function should return 0.

        The original table is not modified.

        Parameters
        ----------
        comparator : Callable[[Row, Row], int]
            The function used to compare two rows.

        Returns
        -------
        new_table : Table
            A new table with sorted rows.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"a": [1, 3, 5], "b": [2, 4, 6] })
        >>> table.sort_rows(lambda row1, row2: 1)
           a  b
        0  1  2
        1  3  4
        2  5  6
        >>> table.sort_rows(lambda row1, row2: -1)
           a  b
        0  5  6
        1  3  4
        2  1  2
        >>> table.sort_rows(lambda row1, row2: 0)
           a  b
        0  1  2
        1  3  4
        2  5  6
        """
        rows = self.to_rows()
        rows.sort(key=functools.cmp_to_key(comparator))
        return Table.from_rows(rows)

    def split_rows(self, percentage_in_first: float) -> tuple[Table, Table]:
        """
        Split the table into two new tables.

        The original table is not modified.

        Parameters
        ----------
        percentage_in_first : float
            The desired size of the first table in percentage to the given table; must be between 0 and 1.

        Returns
        -------
        result : (Table, Table)
            A tuple containing the two resulting tables. The first table has the specified size, the second table
            contains the rest of the data.

        Raises
        ------
        ValueError:
            if the 'percentage_in_first' is not between 0 and 1.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"temperature": [10, 15, 20, 25, 30], "sales": [54, 74, 90, 206, 210]})
        >>> slices = table.split_rows(0.4)
        >>> slices[0]
           temperature  sales
        0           10     54
        1           15     74
        >>> slices[1]
           temperature  sales
        0           20     90
        1           25    206
        2           30    210
        """
        if percentage_in_first < 0 or percentage_in_first > 1:
            raise ValueError("The given percentage is not between 0 and 1")
        if self.number_of_rows == 0:
            return Table(), Table()
        return (
            self.slice_rows(0, round(percentage_in_first * self.number_of_rows)),
            self.slice_rows(round(percentage_in_first * self.number_of_rows)),
        )

    def tag_columns(self, target_name: str, feature_names: list[str] | None = None) -> TaggedTable:
        """
        Return a new `TaggedTable` with columns marked as a target column or feature columns.

        The original table is not modified.

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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table, TaggedTable
        >>> table = Table.from_dict({"item": ["apple", "milk", "beer"], "price": [1.10, 1.19, 1.79], "amount_bought": [74, 72, 51]})
        >>> tagged_table = table.tag_columns(target_name="amount_bought", feature_names=["item", "price"])
        """
        from ._tagged_table import TaggedTable

        return TaggedTable._from_table(self, target_name, feature_names)

    def transform_column(self, name: str, transformer: Callable[[Row], Any]) -> Table:
        """
        Return a new `Table` with the provided column transformed by calling the provided transformer.

        The original table is not modified.

        Returns
        -------
        result : Table
            The table with the transformed column.

        Raises
        ------
        UnknownColumnNameError
            If the column does not exist.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"item": ["apple", "milk", "beer"], "price": [1.00, 1.19, 1.79]})
        >>> table.transform_column("price", lambda row: row.get_value("price") * 100)
            item  price
        0  apple  100.0
        1   milk  119.0
        2   beer  179.0
        """
        if self.has_column(name):
            items: list = [transformer(item) for item in self.to_rows()]
            result: list[Column] = [Column(name, items)]
            return self.replace_column(name, result)
        similar_columns = self._get_similar_columns(name)
        raise UnknownColumnNameError([name], similar_columns)

    def transform_table(self, transformer: TableTransformer) -> Table:
        """
        Return a new `Table` with a learned transformation applied to this table.

        The original table is not modified.

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
        IllegalSchemaModificationError
            If replacing the column would violate an invariant in the subclass.

        Examples
        --------
        >>> from safeds.data.tabular.transformation import OneHotEncoder
        >>> from safeds.data.tabular.containers import Table
        >>> transformer = OneHotEncoder()
        >>> table = Table.from_dict({"fruit": ["apple", "pear", "apple"], "pet": ["dog", "duck", "duck"]})
        >>> transformer = transformer.fit(table, None)
        >>> table.transform_table(transformer)
           fruit__apple  fruit__pear  pet__dog  pet__duck
        0           1.0          0.0       1.0        0.0
        1           0.0          1.0       0.0        1.0
        2           1.0          0.0       0.0        1.0
        """
        return transformer.transform(self)

    def inverse_transform_table(self, transformer: InvertibleTableTransformer) -> Table:
        """
        Return a new `Table` with the inverted transformation applied by the given transformer.

        The original table is not modified.

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
        >>> table = Table.from_dict({"a": ["j", "k", "k"], "b": ["x", "y", "x"]})
        >>> transformer = transformer.fit(table, None)
        >>> transformed_table = transformer.transform(table)
        >>> transformed_table.inverse_transform_table(transformer)
           a  b
        0  j  x
        1  k  y
        2  k  x
        >>> transformer.inverse_transform(transformed_table)
           a  b
        0  j  x
        1  k  y
        2  k  x
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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"temperature": [10, 15, 20, 25, 30], "sales": [54, 74, 90, 206, 210]})
        >>> image = table.plot_correlation_heatmap()
        """
        only_numerical = self.remove_columns_with_non_numerical_values()

        if self.number_of_rows == 0:
            warnings.warn(
                "An empty table has been used. A correlation heatmap on an empty table will show nothing.",
                stacklevel=2,
            )

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        "Attempting to set identical low and high (xlims|ylims) makes transformation singular;"
                        " automatically expanding."
                    ),
                )
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
        else:
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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"temperature": [10, 15, 20, 25, 30], "sales": [54, 74, 90, 206, 210]})
        >>> image = table.plot_lineplot("temperature", "sales")
        """
        if not self.has_column(x_column_name) or not self.has_column(y_column_name):
            similar_columns_x = self._get_similar_columns(x_column_name)
            similar_columns_y = self._get_similar_columns(y_column_name)
            raise UnknownColumnNameError(
                ([x_column_name] if not self.has_column(x_column_name) else [])
                + ([y_column_name] if not self.has_column(y_column_name) else []),
                (similar_columns_x if not self.has_column(x_column_name) else [])
                + (similar_columns_y if not self.has_column(y_column_name) else []),
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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"temperature": [10, 15, 20, 25, 30], "sales": [54, 74, 90, 206, 210]})
        >>> image = table.plot_scatterplot("temperature", "sales")
        """
        if not self.has_column(x_column_name) or not self.has_column(y_column_name):
            similar_columns_x = self._get_similar_columns(x_column_name)
            similar_columns_y = self._get_similar_columns(y_column_name)
            raise UnknownColumnNameError(
                ([x_column_name] if not self.has_column(x_column_name) else [])
                + ([y_column_name] if not self.has_column(y_column_name) else []),
                (similar_columns_x if not self.has_column(x_column_name) else [])
                + (similar_columns_y if not self.has_column(y_column_name) else []),
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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a":[1, 2], "b": [3, 42]})
        >>> image = table.plot_boxplots()
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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [2, 3, 5, 1], "b": [54, 74, 90, 2014]})
        >>> image = table.plot_histograms()
        """
        col_wrap = min(self.number_of_columns, 3)

        data = pd.melt(self._data.applymap(lambda value: str(value)), value_vars=self.column_names)
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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.to_csv_file("./src/resources/to_csv_file.csv")
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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.to_excel_file("./src/resources/to_excel_file.xlsx")
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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.to_json_file("./src/resources/to_json_file.json")
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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> row1 = Row({"a": 1, "b": 5})
        >>> row2 = Row({"a": 2, "b": 6})
        >>> table1 = Table.from_rows([row1, row2])
        >>> table2 = Table.from_dict({"a": [1, 2], "b": [5, 6]})
        >>> table1 == table2
        True
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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"a":[1, 2],"b":[20, 30]})
        >>> table.to_columns()
        [Column('a', [1, 2]), Column('b', [20, 30])]
        """
        return [self.get_column(name) for name in self._schema.column_names]

    def to_rows(self) -> list[Row]:
        """
        Return a list of the rows.

        Returns
        -------
        rows : list[Row]
            List of rows.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table.from_dict({"a":[1, 2],"b":[20, 30]})
        >>> table.to_rows()
        [Row({
            'a': 1,
            'b': 20
        }), Row({
            'a': 2,
            'b': 30
        })]
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

    # ------------------------------------------------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------------------------------------------------

    def _copy(self) -> Table:
        """
        Return a copy of this table.

        Returns
        -------
        table : Table
            The copy of this table.
        """
        return copy.deepcopy(self)
