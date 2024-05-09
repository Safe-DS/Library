from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from safeds._config._polars import _get_polars_config
from safeds._utils import _check_and_normalize_file_path, _structural_hash
from safeds._utils._random import _get_random_seed
from safeds.data.labeled.containers import ExperimentalTabularDataset
from safeds.data.tabular.plotting._experimental_table_plotter import ExperimentalTablePlotter
from safeds.data.tabular.typing._experimental_polars_data_type import _PolarsDataType
from safeds.data.tabular.typing._experimental_polars_schema import _PolarsSchema
from safeds.exceptions import (
    ClosedBound,
    ColumnLengthMismatchError,
    DuplicateColumnNameError,
    OutOfBoundsError,
    UnknownColumnNameError,
)

from ._experimental_column import ExperimentalColumn
from ._experimental_lazy_cell import _LazyCell
from ._experimental_lazy_vectorized_row import _LazyVectorizedRow
from ._table import Table

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    import polars as pl

    from safeds.data.tabular.transformation import (
        ExperimentalInvertibleTableTransformer,
        ExperimentalTableTransformer,
    )
    from safeds.data.tabular.typing import ExperimentalSchema
    from safeds.data.tabular.typing._experimental_data_type import ExperimentalDataType

    from ._experimental_cell import ExperimentalCell
    from ._experimental_row import ExperimentalRow


class ExperimentalTable:
    """
    A two-dimensional collection of data. It can either be seen as a list of rows or as a list of columns.

    To create a `Table` call the constructor or use one of the following static methods:

    | Method                                                                                                      | Description                            |
    | ----------------------------------------------------------------------------------------------------------- | -------------------------------------- |
    | [from_csv_file][safeds.data.tabular.containers._experimental_table.ExperimentalTable.from_csv_file]         | Create a table from a CSV file.        |
    | [from_json_file][safeds.data.tabular.containers._experimental_table.ExperimentalTable.from_json_file]       | Create a table from a JSON file.       |
    | [from_parquet_file][safeds.data.tabular.containers._experimental_table.ExperimentalTable.from_parquet_file] | Create a table from a Parquet file.    |
    | [from_columns][safeds.data.tabular.containers._experimental_table.ExperimentalTable.from_columns]           | Create a table from a list of columns. |
    | [from_dict][safeds.data.tabular.containers._experimental_table.ExperimentalTable.from_dict]                 | Create a table from a dictionary.      |

    Parameters
    ----------
    data:
        The data of the table. If None, an empty table is created.

    Raises
    ------
    ValueError
        If columns have different lengths.

    Examples
    --------
    >>> from safeds.data.tabular.containers import ExperimentalTable
    >>> table = ExperimentalTable({"a": [1, 2, 3], "b": [4, 5, 6]})
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def from_columns(columns: ExperimentalColumn | list[ExperimentalColumn]) -> ExperimentalTable:
        """
        Create a table from a list of columns.

        Parameters
        ----------
        columns:
            The columns.

        Returns
        -------
        table:
            The created table.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalColumn, ExperimentalTable
        >>> a = ExperimentalColumn("a", [1, 2, 3])
        >>> b = ExperimentalColumn("b", [4, 5, 6])
        >>> ExperimentalTable.from_columns([a, b])
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   1 |   4 |
        |   2 |   5 |
        |   3 |   6 |
        +-----+-----+
        """
        import polars as pl

        # TODO: raises

        if isinstance(columns, ExperimentalColumn):
            columns = [columns]

        return ExperimentalTable._from_polars_lazy_frame(
            pl.LazyFrame([column._series for column in columns]),
        )

    @staticmethod
    def from_csv_file(path: str | Path) -> ExperimentalTable:
        """
        Create a table from a CSV file.

        Parameters
        ----------
        path:
            The path to the CSV file. If the file extension is omitted, it is assumed to be ".csv".

        Returns
        -------
        table:
            The created table.

        Raises
        ------
        FileNotFoundError
            If no file exists at the given path.
        ValueError
            If the path has an extension that is not ".csv".

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> ExperimentalTable.from_csv_file("./src/resources/from_csv_file.csv")
        +-----+-----+-----+
        |   a |   b |   c |
        | --- | --- | --- |
        | i64 | i64 | i64 |
        +=================+
        |   1 |   2 |   1 |
        |   0 |   0 |   7 |
        +-----+-----+-----+
        """
        import polars as pl

        path = _check_and_normalize_file_path(path, ".csv", [".csv"], check_if_file_exists=True)
        return ExperimentalTable._from_polars_lazy_frame(pl.scan_csv(path))

    @staticmethod
    def from_dict(data: dict[str, list[Any]]) -> ExperimentalTable:
        """
        Create a table from a dictionary that maps column names to column values.

        Parameters
        ----------
        data:
            The data.

        Returns
        -------
        table:
            The generated table.

        Raises
        ------
        ValueError
            If columns have different lengths.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        >>> ExperimentalTable.from_dict(data)
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   1 |   4 |
        |   2 |   5 |
        |   3 |   6 |
        +-----+-----+
        """
        return ExperimentalTable(data)

    @staticmethod
    def from_json_file(path: str | Path) -> ExperimentalTable:
        """
        Create a table from a JSON file.

        Parameters
        ----------
        path:
            The path to the JSON file. If the file extension is omitted, it is assumed to be ".json".

        Returns
        -------
        table:
            The created table.

        Raises
        ------
        FileNotFoundError
            If no file exists at the given path.
        ValueError
            If the path has an extension that is not ".json".

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> ExperimentalTable.from_json_file("./src/resources/from_json_file_2.json")
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   1 |   4 |
        |   2 |   5 |
        |   3 |   6 |
        +-----+-----+
        """
        import polars as pl

        path = _check_and_normalize_file_path(path, ".json", [".json"], check_if_file_exists=True)
        return ExperimentalTable._from_polars_data_frame(pl.read_json(path))

    @staticmethod
    def from_parquet_file(path: str | Path) -> ExperimentalTable:
        """
        Create a table from a Parquet file.

        Parameters
        ----------
        path:
            The path to the Parquet file. If the file extension is omitted, it is assumed to be ".parquet".

        Returns
        -------
        table:
            The created table.

        Raises
        ------
        FileNotFoundError
            If no file exists at the given path.
        ValueError
            If the path has an extension that is not ".parquet".

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> ExperimentalTable.from_parquet_file("./src/resources/from_parquet_file.parquet")
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   1 |   4 |
        |   2 |   5 |
        |   3 |   6 |
        +-----+-----+
        """
        import polars as pl

        path = _check_and_normalize_file_path(path, ".parquet", [".parquet"], check_if_file_exists=True)
        return ExperimentalTable._from_polars_lazy_frame(pl.scan_parquet(path))

    @staticmethod
    def _from_polars_data_frame(data: pl.DataFrame) -> ExperimentalTable:
        result = object.__new__(ExperimentalTable)
        result._lazy_frame = data.lazy()
        result.__data_frame_cache = data
        return result

    @staticmethod
    def _from_polars_lazy_frame(data: pl.LazyFrame) -> ExperimentalTable:
        result = object.__new__(ExperimentalTable)
        result._lazy_frame = data
        result.__data_frame_cache = None
        return result

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, data: Mapping[str, Sequence[Any]] | None = None) -> None:
        import polars as pl

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
        self._lazy_frame: pl.LazyFrame = pl.LazyFrame(data)
        self.__data_frame_cache: pl.DataFrame | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExperimentalTable):
            return False
        if self is other:
            return True

        return self._data_frame.frame_equal(other._data_frame)

    def __hash__(self) -> int:
        return _structural_hash(self.schema, self.number_of_rows)

    def __repr__(self) -> str:
        with _get_polars_config():
            return self._data_frame.__repr__()

    def __sizeof__(self) -> int:
        return self._data_frame.estimated_size()

    def __str__(self) -> str:
        with _get_polars_config():
            return self._data_frame.__str__()

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def _data_frame(self) -> pl.DataFrame:
        if self.__data_frame_cache is None:
            self.__data_frame_cache = self._lazy_frame.collect()

        return self.__data_frame_cache

    @property
    def column_names(self) -> list[str]:
        """
        The names of the columns in the table.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.column_names
        ['a', 'b']
        """
        return self._lazy_frame.columns

    @property
    def number_of_columns(self) -> int:
        """
        The number of columns in the table.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.number_of_columns
        2
        """
        return self._lazy_frame.width

    @property
    def number_of_rows(self) -> int:
        """
        The number of rows in the table.

        **Note:** This operation must fully load the data into memory, which can be expensive.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.number_of_rows
        3
        """
        return self._data_frame.height

    @property
    def plot(self) -> ExperimentalTablePlotter:
        """The plotter for the table."""
        return ExperimentalTablePlotter(self)

    @property
    def schema(self) -> ExperimentalSchema:
        """The schema of the table."""
        return _PolarsSchema(self._lazy_frame.schema)

    # ------------------------------------------------------------------------------------------------------------------
    # Column operations
    # ------------------------------------------------------------------------------------------------------------------

    def add_columns(
        self,
        columns: ExperimentalColumn | list[ExperimentalColumn],
    ) -> ExperimentalTable:
        """
        Return a new table with additional columns.

        **Notes:**

        * The original table is not modified.
        * This operation must fully load the data into memory, which can be expensive.

        Parameters
        ----------
        columns:
            The columns to add.

        Returns
        -------
        new_table:
            The table with the additional columns.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalColumn, ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 3]})
        >>> new_column = ExperimentalColumn("b", [4, 5, 6])
        >>> table.add_columns(new_column)
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   1 |   4 |
        |   2 |   5 |
        |   3 |   6 |
        +-----+-----+
        """
        if isinstance(columns, ExperimentalColumn):
            columns = [columns]

        if len(columns) == 0:
            return self

        return ExperimentalTable._from_polars_data_frame(
            self._data_frame.hstack([column._series for column in columns]),
        )

    def add_computed_column(
        self,
        name: str,
        computer: Callable[[ExperimentalRow], ExperimentalCell],
    ) -> ExperimentalTable:
        """
        Return a new table with an additional computed column.

        **Note:** The original table is not modified.

        Parameters
        ----------
        name:
            The name of the new column.
        computer:
            The function that computes the values of the new column.

        Returns
        -------
        new_table:
            The table with the computed column.

        Raises
        ------
        ValueError
            If the column name already exists.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.add_computed_column("c", lambda row: row.get_value("a") + row.get_value("b"))
        +-----+-----+-----+
        |   a |   b |   c |
        | --- | --- | --- |
        | i64 | i64 | i64 |
        +=================+
        |   1 |   4 |   5 |
        |   2 |   5 |   7 |
        |   3 |   6 |   9 |
        +-----+-----+-----+
        """
        if self.has_column(name):
            raise DuplicateColumnNameError(name)

        computed_column = computer(_LazyVectorizedRow(self))

        return self._from_polars_lazy_frame(
            self._lazy_frame.with_columns(computed_column._polars_expression.alias(name)),
        )

    def get_column(self, name: str) -> ExperimentalColumn:
        """
        Get a column from the table.

        **Note:** This operation must fully load the data into memory, which can be expensive.

        Parameters
        ----------
        name:
            The name of the column.

        Returns
        -------
        column:
            The column.

        Raises
        ------
        KeyError
            If the column does not exist.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.get_column("a")
        +-----+
        |   a |
        | --- |
        | i64 |
        +=====+
        |   1 |
        |   2 |
        |   3 |
        +-----+
        """
        if not self.has_column(name):
            raise UnknownColumnNameError([name])

        return ExperimentalColumn._from_polars_series(self._data_frame.get_column(name))

    def get_column_type(self, name: str) -> ExperimentalDataType:
        """
        Get the data type of a column.

        Parameters
        ----------
        name:
            The name of the column.

        Returns
        -------
        type:
            The data type of the column.

        Raises
        ------
        KeyError
            If the column does not exist.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.get_column_type("a")
        Int64
        """
        if not self.has_column(name):
            raise UnknownColumnNameError([name])

        return _PolarsDataType(self._lazy_frame.schema[name])

    def has_column(self, name: str) -> bool:
        """
        Check if the table has a column with a specific name.

        Parameters
        ----------
        name:
            The name of the column.

        Returns
        -------
        has_column:
            Whether the table has a column with the specified name.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.has_column("a")
        True
        """
        return name in self.column_names

    def remove_columns(
        self,
        names: str | list[str],
        /,
    ) -> ExperimentalTable:
        """
        Return a new table without the specified columns.

        **Note:** The original table is not modified.

        Parameters
        ----------
        names:
            The names of the columns to remove.

        Returns
        -------
        new_table:
            The table with the columns removed.

        Raises
        ------
        KeyError
            If a column does not exist.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.remove_columns("a")
        +-----+
        |   b |
        | --- |
        | i64 |
        +=====+
        |   4 |
        |   5 |
        |   6 |
        +-----+
        """
        if isinstance(names, str):
            names = [names]

        # TODO: raises?

        return ExperimentalTable._from_polars_lazy_frame(
            self._lazy_frame.drop(names),
        )

    def remove_columns_except(
        self,
        names: str | list[str],
        /,
    ) -> ExperimentalTable:
        """
        Return a new table with only the specified columns.

        Parameters
        ----------
        names:
            The names of the columns to keep.

        Returns
        -------
        new_table:
            The table with only the specified columns.

        Raises
        ------
        KeyError
            If a column does not exist.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.remove_columns_except("a")
        +-----+
        |   a |
        | --- |
        | i64 |
        +=====+
        |   1 |
        |   2 |
        |   3 |
        +-----+
        """
        if isinstance(names, str):
            names = [names]

        # TODO: raises?

        return ExperimentalTable._from_polars_lazy_frame(
            self._lazy_frame.select(names),
        )

    def remove_columns_with_missing_values(self) -> ExperimentalTable:
        """
        Return a new table without columns that contain missing values.

        **Notes:**

        * The original table is not modified.
        * This operation must fully load the data into memory, which can be expensive.

        Returns
        -------
        new_table:
            The table without columns containing missing values.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 3], "b": [4, 5, None]})
        >>> table.remove_columns_with_missing_values()
        +-----+
        |   a |
        | --- |
        | i64 |
        +=====+
        |   1 |
        |   2 |
        |   3 |
        +-----+
        """
        import polars as pl

        return ExperimentalTable._from_polars_lazy_frame(
            pl.LazyFrame(
                [series for series in self._data_frame.get_columns() if series.null_count() == 0],
            ),
        )

    def remove_non_numeric_columns(self) -> ExperimentalTable:
        """
        Return a new table without non-numeric columns.

        **Note:** The original table is not modified.

        Returns
        -------
        new_table:
            The table without non-numeric columns.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 3], "b": ["4", "5", "6"]})
        >>> table.remove_non_numeric_columns()
        +-----+
        |   a |
        | --- |
        | i64 |
        +=====+
        |   1 |
        |   2 |
        |   3 |
        +-----+
        """
        import polars.selectors as cs

        return ExperimentalTable._from_polars_lazy_frame(
            self._lazy_frame.select(cs.numeric()),
        )

    def rename_column(self, old_name: str, new_name: str) -> ExperimentalTable:
        """
        Return a new table with a column renamed.

        **Note:** The original table is not modified.

        Parameters
        ----------
        old_name:
            The name of the column to rename.
        new_name:
            The new name of the column.

        Returns
        -------
        new_table:
            The table with the column renamed.

        Raises
        ------
        KeyError
            If no column with the old name exists.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.rename_column("a", "c")
        +-----+-----+
        |   c |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   1 |   4 |
        |   2 |   5 |
        |   3 |   6 |
        +-----+-----+
        """
        if not self.has_column(old_name):
            raise UnknownColumnNameError([old_name])

        return ExperimentalTable._from_polars_lazy_frame(
            self._lazy_frame.rename({old_name: new_name}),
        )

    def replace_column(
        self,
        old_name: str,
        new_columns: ExperimentalColumn | list[ExperimentalColumn],
    ) -> ExperimentalTable:
        """
        Return a new table with a column replaced by zero or more columns.

        **Note:** The original table is not modified.

        Parameters
        ----------
        old_name:
            The name of the column to replace.
        new_columns:
            The new column or columns.

        Returns
        -------
        new_table:
            The table with the column replaced.

        Raises
        ------
        KeyError
            If no column with the old name exists.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalColumn, ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.replace_column("a", [])
        +-----+
        |   b |
        | --- |
        | i64 |
        +=====+
        |   4 |
        |   5 |
        |   6 |
        +-----+

        >>> column1 = ExperimentalColumn("c", [7, 8, 9])
        >>> table.replace_column("a", column1)
        +-----+-----+
        |   c |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   7 |   4 |
        |   8 |   5 |
        |   9 |   6 |
        +-----+-----+

        >>> column2 = ExperimentalColumn("d", [10, 11, 12])
        >>> table.replace_column("a", [column1, column2])
        +-----+-----+-----+
        |   c |   d |   b |
        | --- | --- | --- |
        | i64 | i64 | i64 |
        +=================+
        |   7 |  10 |   4 |
        |   8 |  11 |   5 |
        |   9 |  12 |   6 |
        +-----+-----+-----+
        """
        if not self.has_column(old_name):
            raise UnknownColumnNameError([old_name])

        if isinstance(new_columns, ExperimentalColumn):
            new_columns = [new_columns]

        if len(new_columns) == 0:
            return self.remove_columns(old_name)

        if len(new_columns) == 1:
            new_column = new_columns[0]
            return ExperimentalTable._from_polars_lazy_frame(
                self._lazy_frame.with_columns(new_column._series.alias(old_name)).rename({old_name: new_column.name}),
            )

        import polars as pl

        index = self.column_names.index(old_name)

        return ExperimentalTable._from_polars_lazy_frame(
            self._lazy_frame.select(
                *[pl.col(name) for name in self.column_names[:index]],
                *[column._series for column in new_columns],
                *[pl.col(name) for name in self.column_names[index + 1 :]],
            ),
        )

    def transform_column(
        self,
        name: str,
        transformer: Callable[[ExperimentalCell], ExperimentalCell],
    ) -> ExperimentalTable:
        """
        Return a new table with a column transformed.

        **Note:** The original table is not modified.

        Parameters
        ----------
        name:
            The name of the column to transform.

        transformer:
            The function that transforms the column.

        Returns
        -------
        new_table:
            The table with the transformed column.

        Raises
        ------
        KeyError
            If no column with the specified name exists.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.transform_column("a", lambda cell: cell + 1)
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   2 |   4 |
        |   3 |   5 |
        |   4 |   6 |
        +-----+-----+
        """
        if not self.has_column(name):
            raise UnknownColumnNameError([name])  # TODO: in the error, compute similar column names

        import polars as pl

        transformed_column = transformer(_LazyCell(pl.col(name)))

        return ExperimentalTable._from_polars_lazy_frame(
            self._lazy_frame.with_columns(transformed_column._polars_expression),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Row operations
    # ------------------------------------------------------------------------------------------------------------------

    # TODO: Rethink group_rows/group_rows_by_column. They should not return a dict.

    def remove_duplicate_rows(self) -> ExperimentalTable:
        """
        Return a new table without duplicate rows.

        **Note:** The original table is not modified.

        Returns
        -------
        new_table:
            The table without duplicate rows.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 2], "b": [4, 5, 5]})
        >>> table.remove_duplicate_rows()
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   1 |   4 |
        |   2 |   5 |
        +-----+-----+
        """
        return ExperimentalTable._from_polars_lazy_frame(
            self._lazy_frame.unique(maintain_order=True),
        )

    def remove_rows(
        self,
        query: Callable[[ExperimentalRow], ExperimentalCell[bool]],
    ) -> ExperimentalTable:
        """
        Return a new table without rows that satisfy a condition.

        **Note:** The original table is not modified.

        Parameters
        ----------
        query:
            The function that determines which rows to remove.

        Returns
        -------
        new_table:
            The table without the specified rows.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.remove_rows(lambda row: row.get_value("a") == 2)
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   1 |   4 |
        |   3 |   6 |
        +-----+-----+
        """
        mask = query(_LazyVectorizedRow(self))

        return ExperimentalTable._from_polars_lazy_frame(
            self._lazy_frame.filter(~mask._polars_expression),
        )

    def remove_rows_by_column(
        self,
        name: str,
        query: Callable[[ExperimentalCell], ExperimentalCell[bool]],
    ) -> ExperimentalTable:
        """
        Return a new table without rows that satisfy a condition on a specific column.

        **Note:** The original table is not modified.

        Parameters
        ----------
        name:
            The name of the column.
        query:
            The function that determines which rows to remove.

        Returns
        -------
        new_table:
            The table without the specified rows.

        Raises
        ------
        KeyError
            If the column does not exist.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.remove_rows_by_column("a", lambda cell: cell == 2)
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   1 |   4 |
        |   3 |   6 |
        +-----+-----+
        """
        import polars as pl

        if not self.has_column(name):
            raise UnknownColumnNameError([name])

        mask = query(_LazyCell(pl.col(name)))

        return ExperimentalTable._from_polars_lazy_frame(
            self._lazy_frame.filter(~mask._polars_expression),
        )

    def remove_rows_with_missing_values(
        self,
        column_names: list[str] | None = None,
    ) -> ExperimentalTable:
        """
        Return a new table without rows containing missing values in the specified columns.

        **Note:** The original table is not modified.

        Parameters
        ----------
        column_names:
            Names of the columns to consider. If None, all columns are considered.

        Returns
        -------
        new_table:
            The table without rows containing missing values in the specified columns.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, None, 3], "b": [4, 5, None]})
        >>> table.remove_rows_with_missing_values()
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   1 |   4 |
        +-----+-----+
        """
        return ExperimentalTable._from_polars_lazy_frame(
            self._lazy_frame.drop_nulls(subset=column_names),
        )

    def remove_rows_with_outliers(
        self,
        column_names: list[str] | None = None,
        *,
        z_score_threshold: float = 3,
    ) -> ExperimentalTable:
        """
        Return a new table without rows containing outliers in the specified columns.

        Whether a data point is an outlier in a column is determined by its z-score. The z-score the distance of the
        data point from the mean of the column divided by the standard deviation of the column. If the z-score is
        greater than the given threshold, the data point is considered an outlier. Missing values are ignored during the
        calculation of the z-score.

        The z-score is only defined for numeric columns. Non-numeric columns are ignored, even if they are specified in
        `column_names`.

        **Notes:**

        * The original table is not modified.
        * This operation must fully load the data into memory, which can be expensive.

        Parameters
        ----------
        column_names:
            Names of the columns to consider. If None, all numeric columns are considered.
        z_score_threshold:
            The z-score threshold for detecting outliers.

        Returns
        -------
        new_table:
            The table without rows containing outliers in the specified columns.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable(
        ...     {
        ...         "a": [1, 2, 3, 4, 5, 6, 1000, None],
        ...         "b": [1, 2, 3, 4, 5, 6,    7,    8],
        ...     }
        ... )
        >>> table.remove_rows_with_outliers(z_score_threshold=2)
        +------+-----+
        |    a |   b |
        |  --- | --- |
        |  i64 | i64 |
        +============+
        |    1 |   1 |
        |    2 |   2 |
        |    3 |   3 |
        |    4 |   4 |
        |    5 |   5 |
        |    6 |   6 |
        | null |   8 |
        +------+-----+
        """
        if column_names is None:
            column_names = self.column_names

        import polars as pl
        import polars.selectors as cs

        non_outlier_mask = pl.all_horizontal(
            self._data_frame.select(cs.numeric() & cs.by_name(column_names)).select(
                pl.all().is_null() | (((pl.all() - pl.all().mean()) / pl.all().std()).abs() <= z_score_threshold),
            ),
        )

        return ExperimentalTable._from_polars_lazy_frame(
            self._lazy_frame.filter(non_outlier_mask),
        )

    def shuffle_rows(self) -> ExperimentalTable:
        """
        Return a new table with the rows shuffled.

        **Note:** The original table is not modified.

        Returns
        -------
        new_table:
            The table with the rows shuffled.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.shuffle_rows()
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   3 |   6 |
        |   2 |   5 |
        |   1 |   4 |
        +-----+-----+
        """
        return ExperimentalTable._from_polars_data_frame(
            self._data_frame.sample(
                fraction=1,
                shuffle=True,
                seed=_get_random_seed(),
            ),
        )

    def slice_rows(self, start: int = 0, length: int | None = None) -> ExperimentalTable:
        """
        Return a new table with a slice of rows.

        **Note:** The original table is not modified.

        Parameters
        ----------
        start:
            The start index of the slice.
        length:
            The length of the slice. If None, the slice contains all rows starting from `start`.

        Returns
        -------
        new_table:
            The table with the slice of rows.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.slice_rows(start=1)
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   2 |   5 |
        |   3 |   6 |
        +-----+-----+

        >>> table.slice_rows(start=1, length=1)
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   2 |   5 |
        +-----+-----+
        """
        return ExperimentalTable._from_polars_lazy_frame(
            self._lazy_frame.slice(start, length),
        )

    def sort_rows(
        self,
        key_selector: Callable[[ExperimentalRow], ExperimentalCell],
        *,
        descending: bool = False,
    ) -> ExperimentalTable:
        """
        Return a new table with the rows sorted.

        **Note:** The original table is not modified.

        Parameters
        ----------
        key_selector:
            The function that selects the key to sort by.
        descending:
            Whether to sort in descending order.

        Returns
        -------
        new_table:
            The table with the rows sorted.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [2, 1, 3], "b": [1, 1, 2]})
        >>> table.sort_rows(lambda row: row.get_value("a") - row.get_value("b"))
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   1 |   1 |
        |   2 |   1 |
        |   3 |   2 |
        +-----+-----+
        """
        key = key_selector(_LazyVectorizedRow(self))

        return ExperimentalTable._from_polars_lazy_frame(
            self._lazy_frame.sort(
                key._polars_expression,
                descending=descending,
                maintain_order=True,
            ),
        )

    def sort_rows_by_column(
        self,
        name: str,
        *,
        descending: bool = False,
    ) -> ExperimentalTable:
        """
        Return a new table with the rows sorted by a specific column.

        **Note:** The original table is not modified.

        Parameters
        ----------
        name:
            The name of the column to sort by.
        descending:
            Whether to sort in descending order.

        Returns
        -------
        new_table:
            The table with the rows sorted by the specified column.

        Raises
        ------
        KeyError
            If the column does not exist.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [2, 1, 3], "b": [1, 1, 2]})
        >>> table.sort_rows_by_column("a")
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   1 |   1 |
        |   2 |   1 |
        |   3 |   2 |
        +-----+-----+
        """
        if not self.has_column(name):
            raise UnknownColumnNameError([name])

        return ExperimentalTable._from_polars_lazy_frame(
            self._lazy_frame.sort(
                name,
                descending=descending,
                maintain_order=True,
            ),
        )

    def split_rows(
        self,
        percentage_in_first: float,
        *,
        shuffle: bool = True,
    ) -> tuple[ExperimentalTable, ExperimentalTable]:
        """
        Create two tables by splitting the rows of the current table.

        The first table contains a percentage of the rows specified by `percentage_in_first`, and the second table
        contains the remaining rows.

        **Note:** The original table is not modified.

        Parameters
        ----------
        percentage_in_first:
            The percentage of rows to include in the first table. Must be between 0 and 1.
        shuffle:
            Whether to shuffle the rows before splitting.

        Returns
        -------
        first_table:
            The first table.
        second_table:
            The second table.

        Raises
        ------
        ValueError
            If `percentage_in_first` is not between 0 and 1.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]})
        >>> first_table, second_table = table.split_rows(0.6)
        >>> first_table
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   1 |   6 |
        |   4 |   9 |
        |   3 |   8 |
        +-----+-----+
        >>> second_table
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   5 |  10 |
        |   2 |   7 |
        +-----+-----+
        """
        if percentage_in_first < 0 or percentage_in_first > 1:
            raise OutOfBoundsError(
                actual=percentage_in_first,
                name="percentage_in_first",
                lower_bound=ClosedBound(0),
                upper_bound=ClosedBound(1),
            )

        input_table = self.shuffle_rows() if shuffle else self
        number_of_rows_in_first = round(percentage_in_first * input_table.number_of_rows)

        return (
            input_table.slice_rows(length=number_of_rows_in_first),
            input_table.slice_rows(start=number_of_rows_in_first),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Table operations
    # ------------------------------------------------------------------------------------------------------------------

    def add_table_as_columns(self, other: ExperimentalTable) -> ExperimentalTable:
        """
        Return a new table with the columns of another table added.

        **Notes:**

        * The original tables are not modified.
        * This operation must fully load the data into memory, which can be expensive.

        Parameters
        ----------
        other:
            The table to add as columns.

        Returns
        -------
        new_table:
            The table with the columns added.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table1 = ExperimentalTable({"a": [1, 2, 3]})
        >>> table2 = ExperimentalTable({"b": [4, 5, 6]})
        >>> table1.add_table_as_columns(table2)
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   1 |   4 |
        |   2 |   5 |
        |   3 |   6 |
        +-----+-----+
        """
        # TODO: raises?

        return ExperimentalTable._from_polars_data_frame(
            self._data_frame.hstack(other._data_frame),
        )

    def add_table_as_rows(self, other: ExperimentalTable) -> ExperimentalTable:
        """
        Return a new table with the rows of another table added.

        **Notes:**

        * The original tables are not modified.
        * This operation must fully load the data into memory, which can be expensive.

        Parameters
        ----------
        other:
            The table to add as rows.

        Returns
        -------
        new_table:
            The table with the rows added.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table1 = ExperimentalTable({"a": [1, 2, 3]})
        >>> table2 = ExperimentalTable({"a": [4, 5, 6]})
        >>> table1.add_table_as_rows(table2)
        +-----+
        |   a |
        | --- |
        | i64 |
        +=====+
        |   1 |
        |   2 |
        |   3 |
        |   4 |
        |   5 |
        |   6 |
        +-----+
        """
        # TODO: raises?

        return ExperimentalTable._from_polars_data_frame(
            self._data_frame.vstack(other._data_frame),
        )

    def inverse_transform_table(self, fitted_transformer: ExperimentalInvertibleTableTransformer) -> ExperimentalTable:
        """
        Return a new table inverse-transformed by a **fitted, invertible** transformer.

        **Notes:**

        * The original table is not modified.
        * Depending on the transformer, this operation might fully load the data into memory, which can be expensive.

        Parameters
        ----------
        fitted_transformer:
            The fitted, invertible transformer to apply.

        Returns
        -------
        new_table:
            The inverse-transformed table.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> from safeds.data.tabular.transformation import ExperimentalRangeScaler
        >>> table = ExperimentalTable({"a": [1, 2, 3]})
        >>> transformer, transformed_table = ExperimentalRangeScaler(min_=0, max_=1).fit_and_transform(table, ["a"])
        >>> transformed_table.inverse_transform_table(transformer)
        +---------+
        |       a |
        |     --- |
        |     f64 |
        +=========+
        | 1.00000 |
        | 2.00000 |
        | 3.00000 |
        +---------+
        """
        return fitted_transformer.inverse_transform(self)

    def transform_table(self, fitted_transformer: ExperimentalTableTransformer) -> ExperimentalTable:
        """
        Return a new table transformed by a **fitted** transformer.

        **Notes:**

        * The original table is not modified.
        * Depending on the transformer, this operation might fully load the data into memory, which can be expensive.


        Parameters
        ----------
        fitted_transformer:
            The fitted transformer to apply.

        Returns
        -------
        new_table:
            The transformed table.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> from safeds.data.tabular.transformation import ExperimentalRangeScaler
        >>> table = ExperimentalTable({"a": [1, 2, 3]})
        >>> transformer = ExperimentalRangeScaler(min_=0, max_=1).fit(table, ["a"])
        >>> table.transform_table(transformer)
        +---------+
        |       a |
        |     --- |
        |     f64 |
        +=========+
        | 0.00000 |
        | 0.50000 |
        | 1.00000 |
        +---------+
        """
        return fitted_transformer.transform(self)

    # ------------------------------------------------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------------------------------------------------

    def summarize_statistics(self) -> ExperimentalTable:
        """
        Return a table with important statistics about this table.

        Returns
        -------
        statistics:
            The table with statistics.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 3]})
        >>> table.summarize_statistics()
        +----------------------+--------------------+
        | metric               | a                  |
        | ---                  | ---                |
        | str                  | str                |
        +===========================================+
        | min                  | 1                  |
        | max                  | 3                  |
        | mean                 | 2.0                |
        | median               | 2.0                |
        | standard deviation   | 1.4142135623730951 |
        | distinct value count | 2                  |
        | idness               | 1.0                |
        | missing value ratio  | 0.0                |
        | stability            | 0.5                |
        +----------------------+--------------------+
        """
        if self.number_of_columns == 0:
            return ExperimentalTable()

        head = self.get_column(self.column_names[0]).summarize_statistics()
        tail = [self.get_column(name).summarize_statistics().get_column(name)._series for name in self.column_names[1:]]

        return ExperimentalTable._from_polars_data_frame(
            head._lazy_frame.collect().hstack(tail, in_place=True),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------------------------------------------------------

    def to_columns(self) -> list[ExperimentalColumn]:
        """
        Return the data of the table as a list of columns.

        Returns
        -------
        columns:
            List of columns.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> columns = table.to_columns()
        """
        return [ExperimentalColumn._from_polars_series(column) for column in self._data_frame.get_columns()]

    def to_csv_file(self, path: str | Path) -> None:
        """
        Write the table to a CSV file.

        If the file and/or the parent directories do not exist, they will be created. If the file exists already, it
        will be overwritten.

        Parameters
        ----------
        path:
            The path to the CSV file. If the file extension is omitted, it is assumed to be ".csv".

        Raises
        ------
        ValueError
            If the path has an extension that is not ".csv".

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.to_csv_file("./src/resources/to_csv_file.csv")
        """
        path = _check_and_normalize_file_path(path, ".csv", [".csv"])
        path.parent.mkdir(parents=True, exist_ok=True)

        self._lazy_frame.sink_csv(path)

    def to_dict(self) -> dict[str, list[Any]]:
        """
        Return a dictionary that maps column names to column values.

        Returns
        -------
        dict_:
            Dictionary representation of the table.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.to_dict()
        {'a': [1, 2, 3], 'b': [4, 5, 6]}
        """
        return self._data_frame.to_dict(as_series=False)

    def to_json_file(
        self,
        path: str | Path,
        *,
        orientation: Literal["column", "row"] = "column",
    ) -> None:
        """
        Write the table to a JSON file.

        If the file and/or the parent directories do not exist, they will be created. If the file exists already, it
        will be overwritten.

        **Note:** This operation must fully load the data into memory, which can be expensive.

        Parameters
        ----------
        path:
            The path to the JSON file. If the file extension is omitted, it is assumed to be ".json".
        orientation:
            The orientation of the JSON file. If "column", the JSON file will be structured as a list of columns. If
            "row", the JSON file will be structured as a list of rows. Row orientation is more human-readable, but
            slower and less memory-efficient.

        Raises
        ------
        ValueError
            If the path has an extension that is not ".json".

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.to_json_file("./src/resources/to_json_file_2.json")
        """
        path = _check_and_normalize_file_path(path, ".json", [".json"])
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write JSON to file
        self._data_frame.write_json(path, row_oriented=(orientation == "row"))

    def to_parquet_file(self, path: str | Path) -> None:
        """
        Write the table to a Parquet file.

        If the file and/or the parent directories do not exist, they will be created. If the file exists already, it
        will be overwritten.

        Parameters
        ----------
        path:
            The path to the Parquet file. If the file extension is omitted, it is assumed to be ".parquet".

        Raises
        ------
        ValueError
            If the path has an extension that is not ".parquet".

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.to_parquet_file("./src/resources/to_parquet_file.parquet")
        """
        path = _check_and_normalize_file_path(path, ".parquet", [".parquet"])
        path.parent.mkdir(parents=True, exist_ok=True)

        self._lazy_frame.sink_parquet(path)

    def to_tabular_dataset(self, target_name: str, extra_names: list[str] | None = None) -> ExperimentalTabularDataset:
        """
        Return a new `TabularDataset` with columns marked as a target, feature, or extra.

        * The target column is the column that a model should predict.
        * Feature columns are columns that a model should use to make predictions.
        * Extra columns are columns that are neither feature nor target. They can be used to provide additional context,
          like an ID column.

        Feature columns are implicitly defined as all columns except the target and extra columns. If no extra columns
        are specified, all columns except the target column are used as features.

        Parameters
        ----------
        target_name:
            Name of the target column.
        extra_names:
            Names of the columns that are neither feature nor target. If None, no extra columns are used, i.e. all but
            the target column are used as features.

        Returns
        -------
        dataset:
            A new tabular dataset with the given target and feature names.

        Raises
        ------
        ValueError
            If the target column is also a feature column.
        ValueError
            If no feature columns are specified.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable(
        ...     {
        ...         "item": ["apple", "milk", "beer"],
        ...         "price": [1.10, 1.19, 1.79],
        ...         "amount_bought": [74, 72, 51],
        ...     }
        ... )
        >>> dataset = table.to_tabular_dataset(target_name="amount_bought", extra_names=["item"])
        """
        return ExperimentalTabularDataset(self, target_name, extra_names)

    def temporary_to_old_table(self) -> Table:
        """
        Convert the table to the old table format. This method is temporary and will be removed in a later version.

        Returns
        -------
        old_table:
            The table in the old format.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> old_table = table.temporary_to_old_table()
        """
        return Table._from_pandas_dataframe(self._data_frame.to_pandas())

    # ------------------------------------------------------------------------------------------------------------------
    # Dataframe interchange protocol
    # ------------------------------------------------------------------------------------------------------------------

    def __dataframe__(self, nan_as_null: bool = False, allow_copy: bool = True):  # type: ignore[no-untyped-def]
        """
        Return a dataframe object that conforms to the dataframe interchange protocol.

        Generally, there is no reason to call this method directly. The dataframe interchange protocol is designed to
        allow libraries to consume tabular data from different sources, such as `pandas` or `polars`. If you still
        decide to call this method, you should not rely on any capabilities of the returned object beyond the dataframe
        interchange protocol.

        The specification of the dataframe interchange protocol can be found
        [here](https://data-apis.org/dataframe-protocol/latest/index.html).

        **Note:** This operation must fully load the data into memory, which can be expensive.

        Parameters
        ----------
        nan_as_null:
            This parameter is deprecated and will be removed in a later revision of the dataframe interchange protocol.
            Setting it has no effect.
        allow_copy:
            Whether memory may be copied to create the dataframe object.

        Returns
        -------
        dataframe:
            A dataframe object that conforms to the dataframe interchange protocol.
        """
        return self._data_frame.__dataframe__(allow_copy=allow_copy)

    # ------------------------------------------------------------------------------------------------------------------
    # IPython integration
    # ------------------------------------------------------------------------------------------------------------------

    def _repr_html_(self) -> str:
        """
        Return a compact HTML representation of the table for IPython.

        **Note:** This operation must fully load the data into memory, which can be expensive.

        Returns
        -------
        html:
            The generated HTML.
        """
        return self._data_frame._repr_html_()
