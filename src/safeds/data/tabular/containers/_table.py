from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

from safeds._config import _get_device, _init_default_device
from safeds._config._polars import _get_polars_config
from safeds._utils import (
    _compute_duplicates,
    _safe_collect_lazy_frame,
    _safe_collect_lazy_frame_schema,
    _structural_hash,
)
from safeds._validation import (
    _check_bounds,
    _check_columns_dont_exist,
    _check_columns_exist,
    _check_row_counts_are_equal,
    _check_schema,
    _ClosedBound,
    _normalize_and_check_file_path,
)
from safeds.data.tabular.plotting import TablePlotter
from safeds.data.tabular.typing import Schema
from safeds.exceptions import (
    DuplicateColumnError,
    LengthMismatchError,
)

from ._column import Column
from ._lazy_cell import _LazyCell
from ._lazy_vectorized_row import _LazyVectorizedRow

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    import polars as pl
    import torch
    from polars.interchange.protocol import DataFrame
    from torch import Tensor
    from torch.utils.data import DataLoader, Dataset

    from safeds.data.labeled.containers import TabularDataset
    from safeds.data.tabular.transformation import (
        InvertibleTableTransformer,
        TableTransformer,
    )
    from safeds.data.tabular.typing import ColumnType
    from safeds.exceptions import (  # noqa: F401
        ColumnNotFoundError,
        ColumnTypeError,
        FileExtensionError,
        NotFittedError,
        NotInvertibleError,
        OutOfBoundsError,
    )

    from ._cell import Cell
    from ._row import Row


class Table:
    """
    A two-dimensional collection of data. It can either be seen as a list of rows or as a list of columns.

    To create a `Table` call the constructor or use one of the following static methods:

    - [`from_csv_file`][safeds.data.tabular.containers._table.Table.from_csv_file]: Create a table from a CSV file.
    - [`from_json_file`][safeds.data.tabular.containers._table.Table.from_json_file]: Create a table from a JSON file.
    - [`from_parquet_file`][safeds.data.tabular.containers._table.Table.from_parquet_file]: Create a table from a Parquet file.
    - [`from_columns`][safeds.data.tabular.containers._table.Table.from_columns]: Create a table from a list of columns.
    - [`from_dict`][safeds.data.tabular.containers._table.Table.from_dict]: Create a table from a dictionary.

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
    >>> from safeds.data.tabular.containers import Table
    >>> Table({"a": [1, 2, 3], "b": [4, 5, 6]})
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

    # ------------------------------------------------------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def from_columns(columns: Column | list[Column]) -> Table:
        """
        Create a table from columns.

        Parameters
        ----------
        columns:
            The columns.

        Returns
        -------
        table:
            The created table.

        Raises
        ------
        LengthMismatchError
            If some columns have different lengths.
        DuplicateColumnError
            If multiple columns have the same name.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column, Table
        >>> a = Column("a", [1, 2, 3])
        >>> b = Column("b", [4, 5, 6])
        >>> Table.from_columns([a, b])
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

        if isinstance(columns, Column):
            columns = [columns]
        if len(columns) == 0:
            return Table({})

        _check_columns_dont_exist(Table({}), [column.name for column in columns])
        _check_row_counts_are_equal(columns)

        return Table._from_polars_lazy_frame(
            pl.concat(
                [column._lazy_frame for column in columns],
                how="horizontal",
            ),
        )

    @staticmethod
    def from_csv_file(path: str | Path, *, separator: str = ",") -> Table:
        """
        Create a table from a CSV file.

        Parameters
        ----------
        path:
            The path to the CSV file. If the file extension is omitted, it is assumed to be ".csv".
        separator:
            The separator between the values in the CSV file.

        Returns
        -------
        table:
            The created table.

        Raises
        ------
        FileExtensionError
            If the path has an extension that is not ".csv".
        FileNotFoundError
            If no file exists at the given path.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> Table.from_csv_file("./src/resources/from_csv_file.csv")
        +-----+-----+-----+
        |   a |   b |   c |
        | --- | --- | --- |
        | i64 | i64 | i64 |
        +=================+
        |   1 |   2 |   1 |
        |   0 |   0 |   7 |
        +-----+-----+-----+

        Related
        -------
        - [`from_json_file`][safeds.data.tabular.containers._table.Table.from_json_file]
        - [`from_parquet_file`][safeds.data.tabular.containers._table.Table.from_parquet_file]
        """
        import polars as pl

        path = _normalize_and_check_file_path(path, ".csv", [".csv"], check_if_file_exists=True)

        return Table._from_polars_lazy_frame(pl.scan_csv(path, separator=separator, raise_if_empty=False))

    @staticmethod
    def from_dict(data: dict[str, list[Any]]) -> Table:
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
        LengthMismatchError
            If columns have different row counts.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        >>> Table.from_dict(data)
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
        return Table(data)

    @staticmethod
    def from_json_file(path: str | Path) -> Table:
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
        FileExtensionError
            If the path has an extension that is not ".json".
        FileNotFoundError
            If no file exists at the given path.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> Table.from_json_file("./src/resources/from_json_file.json")
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   1 |   4 |
        |   2 |   5 |
        |   3 |   6 |
        +-----+-----+

        Related
        -------
        - [`from_csv_file`][safeds.data.tabular.containers._table.Table.from_csv_file]
        - [`from_parquet_file`][safeds.data.tabular.containers._table.Table.from_parquet_file]
        """
        import polars as pl

        path = _normalize_and_check_file_path(path, ".json", [".json"], check_if_file_exists=True)

        return Table._from_polars_data_frame(pl.read_json(path))

    @staticmethod
    def from_parquet_file(path: str | Path) -> Table:
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
        FileExtensionError
            If the path has an extension that is not ".parquet".
        FileNotFoundError
            If no file exists at the given path.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> Table.from_parquet_file("./src/resources/from_parquet_file.parquet")
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   1 |   4 |
        |   2 |   5 |
        |   3 |   6 |
        +-----+-----+

        Related
        -------
        - [`from_csv_file`][safeds.data.tabular.containers._table.Table.from_csv_file]
        - [`from_json_file`][safeds.data.tabular.containers._table.Table.from_json_file]
        """
        import polars as pl

        path = _normalize_and_check_file_path(path, ".parquet", [".parquet"], check_if_file_exists=True)
        return Table._from_polars_lazy_frame(pl.scan_parquet(path))

    @staticmethod
    def _from_polars_data_frame(data: pl.DataFrame) -> Table:
        result = object.__new__(Table)
        result._lazy_frame = data.lazy()
        result.__data_frame_cache = data
        return result

    @staticmethod
    def _from_polars_lazy_frame(data: pl.LazyFrame) -> Table:
        result = object.__new__(Table)
        result._lazy_frame = data
        result.__data_frame_cache = None
        return result

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, data: Mapping[str, Sequence[object]]) -> None:
        import polars as pl

        # Validation
        _check_row_counts_are_equal(data)

        # Implementation
        self._lazy_frame: pl.LazyFrame = pl.LazyFrame(data, strict=False)
        self.__data_frame_cache: pl.DataFrame | None = None  # Scramble the name to prevent access from outside

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Table):
            return NotImplemented
        if self is other:
            return True

        return self._data_frame.equals(other._data_frame)

    def __hash__(self) -> int:
        return _structural_hash(self.schema, self.row_count)

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
            self.__data_frame_cache = _safe_collect_lazy_frame(self._lazy_frame)

        return self.__data_frame_cache

    @property
    def column_count(self) -> int:
        """
        The number of columns.

        **Note:** This operation must compute the schema of the table, which can be expensive.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.column_count
        2
        """
        return len(self.column_names)

    @property
    def column_names(self) -> list[str]:
        """
        The names of the columns in the table.

        **Note:** This operation must compute the schema of the table, which can be expensive.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.column_names
        ['a', 'b']
        """
        return self.schema.column_names

    @property
    def row_count(self) -> int:
        """
        The number of rows.

        **Note:** This operation must fully load the data into memory, which can be expensive.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.row_count
        3
        """
        return self._data_frame.height

    @property
    def plot(self) -> TablePlotter:
        """
        The plotter for the table.

        Call methods of the plotter to create various plots for the table.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> plot = table.plot.box_plots()
        """
        return TablePlotter(self)

    @property
    def schema(self) -> Schema:
        """
        The schema of the table, which is a mapping from column names to their types.

        **Note:** This operation must compute the schema of the table, which can be expensive.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.schema
        Schema({
            'a': int64,
            'b': int64
        })
        """
        return Schema._from_polars_schema(
            _safe_collect_lazy_frame_schema(self._lazy_frame),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Column operations
    # ------------------------------------------------------------------------------------------------------------------

    def add_columns(
        self,
        columns: Column | list[Column] | Table,
    ) -> Table:
        """
        Add columns to the table and return the result as a new table.

        **Note:** The original table is not modified.

        Parameters
        ----------
        columns:
            The columns to add.

        Returns
        -------
        new_table:
            The table with the additional columns.

        Raises
        ------
        DuplicateColumnError
            If a column name exists already. This can also happen if the new columns have duplicate names.
        LengthMismatchError
            If the columns have different row counts.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column, Table
        >>> table = Table({"a": [1, 2, 3]})
        >>> new_column = Column("b", [4, 5, 6])
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

        Related
        -------
        - [`add_computed_column`][safeds.data.tabular.containers._table.Table.add_computed_column]:
            Add a column with values computed from other columns.
        - [`add_index_column`][safeds.data.tabular.containers._table.Table.add_index_column]
        """
        import polars as pl

        if isinstance(columns, Table):
            return self.add_tables_as_columns(columns)

        if isinstance(columns, Column):
            columns = [columns]
        if len(columns) == 0:
            return self

        _check_columns_dont_exist(self, [column.name for column in columns])
        _check_row_counts_are_equal([self, *columns], ignore_entries_without_rows=True)

        return Table._from_polars_lazy_frame(
            pl.concat(
                [
                    self._lazy_frame,
                    *[column._lazy_frame for column in columns],
                ],
                how="horizontal",
            ),
        )

    def add_computed_column(
        self,
        name: str,
        computer: Callable[[Row], Cell],
    ) -> Table:
        """
        Add a computed column to the table and return the result as a new table.

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
        DuplicateColumnError
            If the column name exists already.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.add_computed_column("c", lambda row: row["a"] + row["b"])
        +-----+-----+-----+
        |   a |   b |   c |
        | --- | --- | --- |
        | i64 | i64 | i64 |
        +=================+
        |   1 |   4 |   5 |
        |   2 |   5 |   7 |
        |   3 |   6 |   9 |
        +-----+-----+-----+

        Related
        -------
        - [`add_columns`][safeds.data.tabular.containers._table.Table.add_columns]:
            Add column objects to the table.
        - [`add_index_column`][safeds.data.tabular.containers._table.Table.add_index_column]
        - [`transform_columns`][safeds.data.tabular.containers._table.Table.transform_columns]:
            Transform existing columns with a custom function.
        """
        _check_columns_dont_exist(self, name)

        # When called on a frame without columns, a pl.lit expression adds a single column with a single row
        if self.column_count == 0:
            return self.add_columns(Column(name, []))

        computed_column = computer(_LazyVectorizedRow(self))

        return self._from_polars_lazy_frame(
            self._lazy_frame.with_columns(computed_column._polars_expression.alias(name)),
        )

    def add_index_column(self, name: str, *, first_index: int = 0) -> Table:
        """
        Add an index column to the table and return the result as a new table.

        **Note:** The original table is not modified.

        Parameters
        ----------
        name:
            The name of the new column.
        first_index:
            The index to assign to the first row. Must be greater or equal to 0.

        Returns
        -------
        new_table:
            The table with the index column.

        Raises
        ------
        DuplicateColumnError
            If the column name exists already.
        OutOfBoundsError
            If `first_index` is negative.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.add_index_column("id")
        +-----+-----+-----+
        |  id |   a |   b |
        | --- | --- | --- |
        | u32 | i64 | i64 |
        +=================+
        |   0 |   1 |   4 |
        |   1 |   2 |   5 |
        |   2 |   3 |   6 |
        +-----+-----+-----+

        >>> table.add_index_column("id", first_index=10)
        +-----+-----+-----+
        |  id |   a |   b |
        | --- | --- | --- |
        | u32 | i64 | i64 |
        +=================+
        |  10 |   1 |   4 |
        |  11 |   2 |   5 |
        |  12 |   3 |   6 |
        +-----+-----+-----+

        Related
        -------
        - [`add_columns`][safeds.data.tabular.containers._table.Table.add_columns]:
            Add column objects to the table.
        - [`add_computed_column`][safeds.data.tabular.containers._table.Table.add_computed_column]:
            Add a column with values computed from other columns.
        """
        _check_columns_dont_exist(self, name)
        _check_bounds(
            "first_index",
            first_index,
            lower_bound=_ClosedBound(0),
        )

        return Table._from_polars_lazy_frame(
            self._lazy_frame.with_row_index(name, offset=first_index),
        )

    def get_column(self, name: str) -> Column:
        """
        Get a column from the table.

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
        ColumnNotFoundError
            If the column does not exist.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
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
        _check_columns_exist(self, name)
        return Column._from_polars_lazy_frame(name, self._lazy_frame)

    def get_column_type(self, name: str) -> ColumnType:
        """
        Get the type of a column.

        Parameters
        ----------
        name:
            The name of the column.

        Returns
        -------
        type:
            The type of the column.

        Raises
        ------
        ColumnNotFoundError
            If the column does not exist.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.get_column_type("a")
        int64
        """
        return self.schema.get_column_type(name)

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
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.has_column("a")
        True

        >>> table.has_column("c")
        False
        """
        return self.schema.has_column(name)

    def remove_columns(
        self,
        selector: str | list[str],
        *,
        ignore_unknown_names: bool = False,
    ) -> Table:
        """
        Remove the specified columns from the table and return the result as a new table.

        **Note:** The original table is not modified.

        Parameters
        ----------
        selector:
            The columns to remove.
        ignore_unknown_names:
            If set to True, columns that are not present in the table will be ignored.
            If set to False, an error will be raised if any of the specified columns do not exist.

        Returns
        -------
        new_table:
            The table with the columns removed.

        Raises
        ------
        ColumnNotFoundError
            If a column does not exist and unknown names are not ignored.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
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

        >>> table.remove_columns(["c"], ignore_unknown_names=True)
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   1 |   4 |
        |   2 |   5 |
        |   3 |   6 |
        +-----+-----+

        Related
        -------
        - [`select_columns`][safeds.data.tabular.containers._table.Table.select_columns]:
            Keep only a subset of the columns.
        - [`remove_columns_with_missing_values`][safeds.data.tabular.containers._table.Table.remove_columns_with_missing_values]
        - [`remove_non_numeric_columns`][safeds.data.tabular.containers._table.Table.remove_non_numeric_columns]
        """
        if isinstance(selector, str):
            selector = [selector]

        if not ignore_unknown_names:
            _check_columns_exist(self, selector)

        return Table._from_polars_lazy_frame(
            self._lazy_frame.drop(selector, strict=not ignore_unknown_names),
        )

    def remove_columns_with_missing_values(
        self,
        *,
        missing_value_ratio_threshold: float = 0,
    ) -> Table:
        """
        Remove columns with too many missing values and return the result as a new table.

        How many missing values are allowed is determined by the `missing_value_ratio_threshold` parameter. A column is
        removed if its missing value ratio is greater than the threshold. By default, a column is removed if it contains
        any missing values.

        **Notes:**

        - The original table is not modified.
        - This operation must fully load the data into memory, which can be expensive.

        Parameters
        ----------
        missing_value_ratio_threshold:
            The maximum missing value ratio a column can have to be kept (inclusive). Must be between 0 and 1.

        Returns
        -------
        new_table:
            The table without columns that contain too many missing values.

        Raises
        ------
        OutOfBoundsError
            If the `missing_value_ratio_threshold` is not between 0 and 1.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, None]})
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

        Related
        -------
        - [`remove_rows_with_missing_values`][safeds.data.tabular.containers._table.Table.remove_rows_with_missing_values]
        - [`SimpleImputer`][safeds.data.tabular.transformation._simple_imputer.SimpleImputer]:
            Replace missing values with a constant value or a statistic of the column.
        - [`KNearestNeighborsImputer`][safeds.data.tabular.transformation._k_nearest_neighbors_imputer.KNearestNeighborsImputer]:
            Replace missing values with a value computed from the nearest neighbors.
        - [`select_columns`][safeds.data.tabular.containers._table.Table.select_columns]:
            Keep only a subset of the columns.
        - [`remove_columns`][safeds.data.tabular.containers._table.Table.remove_columns]:
            Remove columns from the table by name.
        - [`remove_non_numeric_columns`][safeds.data.tabular.containers._table.Table.remove_non_numeric_columns]
        """
        import polars as pl

        _check_bounds(
            "max_missing_value_ratio",
            missing_value_ratio_threshold,
            lower_bound=_ClosedBound(0),
            upper_bound=_ClosedBound(1),
        )

        # Collect the data here, since we need it again later
        mask = self._data_frame.select(
            (pl.all().null_count() / pl.len() <= missing_value_ratio_threshold),
        )

        if mask.is_empty():
            return Table({})

        return Table._from_polars_data_frame(
            self._data_frame[:, mask.row(0)],
        )

    def remove_non_numeric_columns(self) -> Table:
        """
        Remove non-numeric columns and return the result as a new table.

        **Note:** The original table is not modified.

        Returns
        -------
        new_table:
            The table without non-numeric columns.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": ["4", "5", "6"]})
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

        Related
        -------
        - [`select_columns`][safeds.data.tabular.containers._table.Table.select_columns]:
            Keep only a subset of the columns.
        - [`remove_columns`][safeds.data.tabular.containers._table.Table.remove_columns]:
            Remove columns from the table by name.
        - [`remove_columns_with_missing_values`][safeds.data.tabular.containers._table.Table.remove_columns_with_missing_values]
        """
        import polars.selectors as cs

        return Table._from_polars_lazy_frame(
            self._lazy_frame.select(cs.numeric()),
        )

    def rename_column(self, old_name: str, new_name: str) -> Table:
        """
        Rename a column and return the result as a new table.

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
        ColumnNotFoundError
            If no column with the old name exists.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
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
        _check_columns_exist(self, old_name)
        _check_columns_dont_exist(self, new_name, old_name=old_name)

        return Table._from_polars_lazy_frame(
            self._lazy_frame.rename({old_name: new_name}),
        )

    def replace_column(
        self,
        old_name: str,
        new_columns: Column | list[Column] | Table,
    ) -> Table:
        """
        Replace a column with zero or more columns and return the result as a new table.

        **Note:** The original table is not modified.

        Parameters
        ----------
        old_name:
            The name of the column to replace.
        new_columns:
            The new columns.

        Returns
        -------
        new_table:
            The table with the column replaced.

        Raises
        ------
        ColumnNotFoundError
            If no column with the old name exists.
        DuplicateColumnError
            If a column name exists already. This can also happen if the new columns have duplicate names.
        LengthMismatchError
            If the columns have different row counts.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column, Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
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

        >>> column1 = Column("c", [7, 8, 9])
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

        >>> column2 = Column("d", [10, 11, 12])
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
        import polars.selectors as cs

        if isinstance(new_columns, Column):
            new_columns = [new_columns]
        elif isinstance(new_columns, Table):
            new_columns = new_columns.to_columns()

        _check_columns_exist(self, old_name)
        _check_columns_dont_exist(self, [column.name for column in new_columns], old_name=old_name)
        _check_row_counts_are_equal([self, *new_columns])

        if len(new_columns) == 0:
            return self.remove_columns(old_name, ignore_unknown_names=True)

        if len(new_columns) == 1:
            new_column = new_columns[0]
            return Table._from_polars_lazy_frame(
                self._lazy_frame.with_columns(new_column._series.alias(old_name)).rename({old_name: new_column.name}),
            )

        column_names = self.column_names
        index = column_names.index(old_name)

        return Table._from_polars_lazy_frame(
            self._lazy_frame.select(
                cs.by_name(column_names[:index]),
                *[column._series for column in new_columns],
                cs.by_name(column_names[index + 1 :]),
            ),
        )

    def select_columns(
        self,
        selector: str | list[str],
    ) -> Table:
        """
        Select a subset of the columns and return the result as a new table.

        **Note:** The original table is not modified.

        Parameters
        ----------
        selector:
            The columns to keep.

        Returns
        -------
        new_table:
            The table with only a subset of the columns.

        Raises
        ------
        ColumnNotFoundError
            If a column does not exist.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.select_columns("a")
        +-----+
        |   a |
        | --- |
        | i64 |
        +=====+
        |   1 |
        |   2 |
        |   3 |
        +-----+

        Related
        -------
        - [`remove_columns`][safeds.data.tabular.containers._table.Table.remove_columns]:
            Remove columns from the table by name.
        - [`remove_columns_with_missing_values`][safeds.data.tabular.containers._table.Table.remove_columns_with_missing_values]
        - [`remove_non_numeric_columns`][safeds.data.tabular.containers._table.Table.remove_non_numeric_columns]
        """
        _check_columns_exist(self, selector)

        return Table._from_polars_lazy_frame(
            self._lazy_frame.select(selector),
        )

    def transform_columns(
        self,
        selector: str | list[str],
        transformer: Callable[[Cell], Cell] | Callable[[Cell, Row], Cell],
    ) -> Table:
        """
        Transform columns with a custom function and return the result as a new table.

        **Note:** The original table is not modified.

        Parameters
        ----------
        selector:
            The names of the columns to transform.
        transformer:
            The function that computes the new values. It may take either a single cell or a cell and the entire row as
            arguments (see examples).

        Returns
        -------
        new_table:
            The table with the transformed column.

        Raises
        ------
        ColumnNotFoundError
            If no column with the specified name exists.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.transform_columns("a", lambda cell: cell + 1)
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   2 |   4 |
        |   3 |   5 |
        |   4 |   6 |
        +-----+-----+

        >>> table.transform_columns(["a", "b"], lambda cell: cell + 1)
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   2 |   5 |
        |   3 |   6 |
        |   4 |   7 |
        +-----+-----+

        >>> table.transform_columns("a", lambda cell, row: cell + row["b"])
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   5 |   4 |
        |   7 |   5 |
        |   9 |   6 |
        +-----+-----+

        Related
        -------
        - [`add_computed_column`][safeds.data.tabular.containers._table.Table.add_computed_column]:
            Add a new column that is computed from other columns.
        - [`transform_table`][safeds.data.tabular.containers._table.Table.transform_table]:
            Transform the entire table with a fitted transformer.
        """
        import polars as pl

        _check_columns_exist(self, selector)

        if isinstance(selector, str):
            selector = [selector]

        parameter_count = transformer.__code__.co_argcount
        if parameter_count == 1:
            # Transformer only takes a cell
            expressions = [
                transformer(  # type: ignore[call-arg]
                    _LazyCell(pl.col(name)),
                )._polars_expression.alias(name)
                for name in selector
            ]
        else:
            # Transformer takes a cell and the entire row
            expressions = [
                transformer(  # type: ignore[call-arg]
                    _LazyCell(pl.col(name)),
                    _LazyVectorizedRow(self),
                )._polars_expression.alias(name)
                for name in selector
            ]

        return Table._from_polars_lazy_frame(
            self._lazy_frame.with_columns(*expressions),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Row operations
    # ------------------------------------------------------------------------------------------------------------------

    @overload
    def count_rows_if(
        self,
        predicate: Callable[[Row], Cell[bool | None]],
        *,
        ignore_unknown: Literal[True] = ...,
    ) -> int: ...

    @overload
    def count_rows_if(
        self,
        predicate: Callable[[Row], Cell[bool | None]],
        *,
        ignore_unknown: bool,
    ) -> int | None: ...

    def count_rows_if(
        self,
        predicate: Callable[[Row], Cell[bool | None]],
        *,
        ignore_unknown: bool = True,
    ) -> int | None:
        """
        Count how many rows in the table satisfy the predicate.

        The predicate can return one of three results:

        * True, if the row satisfies the predicate.
        * False, if the row does not satisfy the predicate.
        * None, if the truthiness of the predicate is unknown, e.g. due to missing values.

        By default, cases where the truthiness of the predicate is unknown are ignored and this method returns how
        often the predicate returns True.

        You can instead enable Kleene logic by setting `ignore_unknown=False`. In this case, this method returns None if
        the predicate returns None at least once. Otherwise, it still returns how often the predicate returns True.

        Parameters
        ----------
        predicate:
            The predicate to apply to each row.
        ignore_unknown:
            Whether to ignore cases where the truthiness of the predicate is unknown.

        Returns
        -------
        count:
            The number of rows in the table that satisfy the predicate.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"col1": [1, 2, 3], "col2": [1, 3, None]})
        >>> table.count_rows_if(lambda row: row["col1"] < row["col2"])
        1

        >>> print(table.count_rows_if(lambda row: row["col1"] < row["col2"], ignore_unknown=False))
        None
        """
        expression = predicate(_LazyVectorizedRow(self))._polars_expression
        series = _safe_collect_lazy_frame(self._lazy_frame.select(expression.alias("count"))).get_column("count")

        if ignore_unknown or series.null_count() == 0:
            return series.sum()
        else:
            return None

    def filter_rows(
        self,
        predicate: Callable[[Row], Cell[bool | None]],
    ) -> Table:
        """
        Keep only rows that satisfy a condition and return the result as a new table.

        **Note:** The original table is not modified.

        Parameters
        ----------
        predicate:
            The function that determines which rows to keep.

        Returns
        -------
        new_table:
            The table containing only the specified rows.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.filter_rows(lambda row: row["a"] == 2)
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   2 |   5 |
        +-----+-----+

        Related
        -------
        - [`filter_rows_by_column`][safeds.data.tabular.containers._table.Table.filter_rows_by_column]:
            Keep only rows that satisfy a condition on a specific column.
        - [`remove_duplicate_rows`][safeds.data.tabular.containers._table.Table.remove_duplicate_rows]
        - [`remove_rows_with_missing_values`][safeds.data.tabular.containers._table.Table.remove_rows_with_missing_values]
        - [`remove_rows_with_outliers`][safeds.data.tabular.containers._table.Table.remove_rows_with_outliers]
        """
        mask = predicate(_LazyVectorizedRow(self))

        return Table._from_polars_lazy_frame(
            self._lazy_frame.filter(mask._polars_expression),
        )

    def filter_rows_by_column(
        self,
        name: str,
        predicate: Callable[[Cell], Cell[bool | None]],
    ) -> Table:
        """
        Keep only rows that satisfy a condition on a specific column and return the result as a new table.

        **Note:** The original table is not modified.

        Parameters
        ----------
        name:
            The name of the column.
        predicate:
            The function that determines which rows to keep.

        Returns
        -------
        new_table:
            The table containing only the specified rows.

        Raises
        ------
        ColumnNotFoundError
            If the column does not exist.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.filter_rows_by_column("a", lambda cell: cell == 2)
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   2 |   5 |
        +-----+-----+

        Related
        -------
        - [`filter_rows`][safeds.data.tabular.containers._table.Table.filter_rows]:
            Keep only rows that satisfy a condition.
        - [`remove_duplicate_rows`][safeds.data.tabular.containers._table.Table.remove_duplicate_rows]
        - [`remove_rows_with_missing_values`][safeds.data.tabular.containers._table.Table.remove_rows_with_missing_values]
        - [`remove_rows_with_outliers`][safeds.data.tabular.containers._table.Table.remove_rows_with_outliers]
        """
        _check_columns_exist(self, name)

        import polars as pl

        mask = predicate(_LazyCell(pl.col(name)))

        return Table._from_polars_lazy_frame(
            self._lazy_frame.filter(mask._polars_expression),
        )

    def remove_duplicate_rows(self) -> Table:
        """
        Remove duplicate rows and return the result as a new table.

        **Note:** The original table is not modified.

        Returns
        -------
        new_table:
            The table without duplicate rows.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 2], "b": [4, 5, 5]})
        >>> table.remove_duplicate_rows()
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   1 |   4 |
        |   2 |   5 |
        +-----+-----+

        Related
        -------
        - [`filter_rows`][safeds.data.tabular.containers._table.Table.filter_rows]:
            Keep only rows that satisfy a condition.
        - [`filter_rows_by_column`][safeds.data.tabular.containers._table.Table.filter_rows_by_column]:
            Keep only rows that satisfy a condition on a specific column.
        - [`remove_rows_with_missing_values`][safeds.data.tabular.containers._table.Table.remove_rows_with_missing_values]
        - [`remove_rows_with_outliers`][safeds.data.tabular.containers._table.Table.remove_rows_with_outliers]
        """
        return Table._from_polars_lazy_frame(
            self._lazy_frame.unique(maintain_order=True),
        )

    def remove_rows(
        self,
        predicate: Callable[[Row], Cell[bool | None]],
    ) -> Table:
        """
        Remove rows that satisfy a condition and return the result as a new table.

        **Note:** The original table is not modified.

        Parameters
        ----------
        predicate:
            The function that determines which rows to remove.

        Returns
        -------
        new_table:
            The table without the specified rows.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.remove_rows(lambda row: row["a"] == 2)
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   1 |   4 |
        |   3 |   6 |
        +-----+-----+

        Related
        -------
        - [`filter_rows`][safeds.data.tabular.containers._table.Table.filter_rows]:
            Keep only rows that satisfy a condition.
        - [`filter_rows_by_column`][safeds.data.tabular.containers._table.Table.filter_rows_by_column]:
            Keep only rows that satisfy a condition on a specific column.
        - [`remove_rows_by_column`][safeds.data.tabular.containers._table.Table.filter_rows_by_column]:
            Remove rows that satisfy a condition on a specific column.
        - [`remove_duplicate_rows`][safeds.data.tabular.containers._table.Table.remove_duplicate_rows]
        - [`remove_rows_with_missing_values`][safeds.data.tabular.containers._table.Table.remove_rows_with_missing_values]
        - [`remove_rows_with_outliers`][safeds.data.tabular.containers._table.Table.remove_rows_with_outliers]
        """
        mask = predicate(_LazyVectorizedRow(self))

        return Table._from_polars_lazy_frame(
            self._lazy_frame.remove(mask._polars_expression),
        )

    def remove_rows_by_column(
        self,
        name: str,
        predicate: Callable[[Cell], Cell[bool | None]],
    ) -> Table:
        """
        Remove rows that satisfy a condition on a specific column and return the result as a new table.

        **Note:** The original table is not modified.

        Parameters
        ----------
        name:
            The name of the column.
        predicate:
            The function that determines which rows to remove.

        Returns
        -------
        new_table:
            The table without the specified rows.

        Raises
        ------
        ColumnNotFoundError
            If the column does not exist.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.remove_rows_by_column("a", lambda cell: cell == 2)
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   1 |   4 |
        |   3 |   6 |
        +-----+-----+

        Related
        -------
        - [`filter_rows`][safeds.data.tabular.containers._table.Table.filter_rows]:
            Keep only rows that satisfy a condition.
        - [`filter_rows_by_column`][safeds.data.tabular.containers._table.Table.filter_rows_by_column]:
            Keep only rows that satisfy a condition on a specific column.
        - [`remove_rows`][safeds.data.tabular.containers._table.Table.remove_rows]:
            Remove rows that satisfy a condition.
        - [`remove_duplicate_rows`][safeds.data.tabular.containers._table.Table.remove_duplicate_rows]
        - [`remove_rows_with_missing_values`][safeds.data.tabular.containers._table.Table.remove_rows_with_missing_values]
        - [`remove_rows_with_outliers`][safeds.data.tabular.containers._table.Table.remove_rows_with_outliers]
        """
        _check_columns_exist(self, name)

        import polars as pl

        mask = predicate(_LazyCell(pl.col(name)))

        return Table._from_polars_lazy_frame(
            self._lazy_frame.remove(mask._polars_expression),
        )

    def remove_rows_with_missing_values(
        self,
        *,
        selector: str | list[str] | None = None,
    ) -> Table:
        """
        Remove rows that contain missing values in the specified columns and return the result as a new table.

        The resulting table no longer has missing values in the specified columns. Be aware that this method can discard
        a lot of data. Consider first removing columns with many missing values, or using one of the imputation methods
        (see "Related" section).

        **Note:** The original table is not modified.

        Parameters
        ----------
        selector:
            The columns to check. If None, all columns are checked.

        Returns
        -------
        new_table:
            The table without rows that contain missing values in the specified columns.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, None, 3], "b": [4, 5, None]})
        >>> table.remove_rows_with_missing_values()
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   1 |   4 |
        +-----+-----+

        >>> table.remove_rows_with_missing_values(selector=["b"])
        +------+-----+
        |    a |   b |
        |  --- | --- |
        |  i64 | i64 |
        +============+
        |    1 |   4 |
        | null |   5 |
        +------+-----+

        Related
        -------
        - [`remove_columns_with_missing_values`][safeds.data.tabular.containers._table.Table.remove_columns_with_missing_values]
        - [`SimpleImputer`][safeds.data.tabular.transformation._simple_imputer.SimpleImputer]:
            Replace missing values with a constant value or a statistic of the column.
        - [`KNearestNeighborsImputer`][safeds.data.tabular.transformation._k_nearest_neighbors_imputer.KNearestNeighborsImputer]:
            Replace missing values with a value computed from the nearest neighbors.
        - [`filter_rows`][safeds.data.tabular.containers._table.Table.filter_rows]:
            Keep only rows that satisfy a condition.
        - [`filter_rows_by_column`][safeds.data.tabular.containers._table.Table.filter_rows_by_column]:
            Keep only rows that satisfy a condition on a specific column.
        - [`remove_duplicate_rows`][safeds.data.tabular.containers._table.Table.remove_duplicate_rows]
        - [`remove_rows_with_outliers`][safeds.data.tabular.containers._table.Table.remove_rows_with_outliers]
        """
        if isinstance(selector, list) and not selector:
            # polars panics in this case
            return self

        return Table._from_polars_lazy_frame(
            self._lazy_frame.drop_nulls(subset=selector),
        )

    def remove_rows_with_outliers(
        self,
        *,
        selector: str | list[str] | None = None,
        z_score_threshold: float = 3,
    ) -> Table:
        """
        Remove rows that contain outliers in the specified columns and return the result as a new table.

        Whether a value is an outlier in a column is determined by its z-score. The z-score the distance of the value
        from the mean of the column divided by the standard deviation of the column. If the z-score is greater than the
        given threshold, the value is considered an outlier. Missing values are ignored during the calculation of the
        z-score.

        The z-score is only defined for numeric columns. Non-numeric columns are ignored, even if they are specified in
        `column_names`.

        **Notes:**

        - The original table is not modified.
        - This operation must fully load the data into memory, which can be expensive.

        Parameters
        ----------
        selector:
            The columns to check. If None, all columns are checked.
        z_score_threshold:
            The z-score threshold for detecting outliers. Must be greater than or equal to 0.

        Returns
        -------
        new_table:
            The table without rows that contain outliers in the specified columns.

        Raises
        ------
        OutOfBoundsError
            If the `z_score_threshold` is less than 0.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table(
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

        Related
        -------
        - [`filter_rows`][safeds.data.tabular.containers._table.Table.filter_rows]:
            Keep only rows that satisfy a condition.
        - [`filter_rows_by_column`][safeds.data.tabular.containers._table.Table.filter_rows_by_column]:
            Keep only rows that satisfy a condition on a specific column.
        - [`remove_duplicate_rows`][safeds.data.tabular.containers._table.Table.remove_duplicate_rows]
        - [`remove_rows_with_missing_values`][safeds.data.tabular.containers._table.Table.remove_rows_with_missing_values]
        """
        _check_bounds(
            "z_score_threshold",
            z_score_threshold,
            lower_bound=_ClosedBound(0),
        )

        if selector is None:
            selector = self.column_names

        import polars as pl
        import polars.selectors as cs

        # polar's `all_horizontal` raises a `ComputeError` if there are no columns
        selected = self._lazy_frame.select(cs.numeric() & cs.by_name(selector))
        selected_names = _safe_collect_lazy_frame_schema(selected).names()
        if not selected_names:
            return self

        # Multiply z-score by standard deviation instead of dividing the distance by it, to avoid division by zero
        non_outlier_mask = pl.all_horizontal(
            _safe_collect_lazy_frame(
                selected.select(
                    pl.all().is_null() | ((pl.all() - pl.all().mean()).abs() <= (z_score_threshold * pl.all().std())),
                ),
            ),
        )

        return Table._from_polars_lazy_frame(
            self._lazy_frame.filter(non_outlier_mask),
        )

    def shuffle_rows(self, *, random_seed: int = 0) -> Table:
        """
        Shuffle the rows and return the result as a new table.

        **Notes:**

        - The original table is not modified.
        - This operation must fully load the data into memory, which can be expensive.

        Parameters
        ----------
        random_seed:
            The seed for the pseudorandom number generator.

        Returns
        -------
        new_table:
            The table with the rows shuffled.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.shuffle_rows()
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   1 |   4 |
        |   3 |   6 |
        |   2 |   5 |
        +-----+-----+
        """
        return Table._from_polars_data_frame(
            self._data_frame.sample(
                fraction=1,
                shuffle=True,
                seed=random_seed,
            ),
        )

    def slice_rows(self, *, start: int = 0, length: int | None = None) -> Table:
        """
        Slice the rows and return the result as a new table.

        **Note:** The original table is not modified.

        Parameters
        ----------
        start:
            The start index of the slice. Nonnegative indices are counted from the beginning (starting at 0), negative
            indices from the end (starting at -1).
        length:
            The length of the slice. If None, the slice contains all rows starting from `start`. Must greater than or
            equal to 0.

        Returns
        -------
        new_table:
            The table with the slice of rows.

        Raises
        ------
        OutOfBoundsError
            If length is less than 0.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
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
        _check_bounds("length", length, lower_bound=_ClosedBound(0))

        return Table._from_polars_lazy_frame(
            self._lazy_frame.slice(start, length),
        )

    def sort_rows(
        self,
        key_selector: Callable[[Row], Cell],
        *,
        descending: bool = False,
    ) -> Table:
        """
        Sort the rows by a custom function and return the result as a new table.

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
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [2, 1, 3], "b": [1, 1, 2]})
        >>> table.sort_rows(lambda row: row["a"] - row["b"])
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   1 |   1 |
        |   2 |   1 |
        |   3 |   2 |
        +-----+-----+

        Related
        -------
        - [`sort_rows_by_column`][safeds.data.tabular.containers._table.Table.sort_rows_by_column]:
            Sort the rows by a specific column.
        """
        key = key_selector(_LazyVectorizedRow(self))

        return Table._from_polars_lazy_frame(
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
    ) -> Table:
        """
        Sort the rows by a specific column and return the result as a new table.

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
        ColumnNotFoundError
            If the column does not exist.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [2, 1, 3], "b": [1, 1, 2]})
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

        Related
        -------
        - [`sort_rows`][safeds.data.tabular.containers._table.Table.sort_rows]:
            Sort the rows by a value computed from an entire row.
        """
        _check_columns_exist(self, name)

        return Table._from_polars_lazy_frame(
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
        random_seed: int = 0,
    ) -> tuple[Table, Table]:
        """
        Create two tables by splitting the rows of the current table.

        The first table contains a percentage of the rows specified by `percentage_in_first`, and the second table
        contains the remaining rows. By default, the rows are shuffled before splitting. You can disable this by setting
        `shuffle` to False.

        **Notes:**

        - The original table is not modified.
        - This operation must fully load the data into memory, which can be expensive.

        Parameters
        ----------
        percentage_in_first:
            The percentage of rows to include in the first table. Must be between 0 and 1.
        shuffle:
            Whether to shuffle the rows before splitting.
        random_seed:
            The seed for the pseudorandom number generator used for shuffling.

        Returns
        -------
        first_table:
            The first table.
        second_table:
            The second table.

        Raises
        ------
        OutOfBoundsError
            If `percentage_in_first` is not between 0 and 1.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]})
        >>> first_table, second_table = table.split_rows(0.6)
        >>> first_table
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   4 |   9 |
        |   1 |   6 |
        |   2 |   7 |
        +-----+-----+
        >>> second_table
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   5 |  10 |
        |   3 |   8 |
        +-----+-----+
        """
        _check_bounds(
            "percentage_in_first",
            percentage_in_first,
            lower_bound=_ClosedBound(0),
            upper_bound=_ClosedBound(1),
        )

        input_table = self.shuffle_rows(random_seed=random_seed) if shuffle else self
        row_count_in_first = round(percentage_in_first * input_table.row_count)

        return (
            input_table.slice_rows(length=row_count_in_first),
            input_table.slice_rows(start=row_count_in_first),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Table operations
    # ------------------------------------------------------------------------------------------------------------------

    def add_tables_as_columns(self, others: Table | list[Table]) -> Table:
        """
        Add the columns of other tables and return the result as a new table.

        **Note:** The original tables are not modified.

        Parameters
        ----------
        others:
            The tables to add as columns.

        Returns
        -------
        new_table:
            The table with the columns added.

        Raises
        ------
        DuplicateColumnError
            If a column name exists already.
        LengthMismatchError
            If the tables have different row counts.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table1 = Table({"a": [1, 2, 3]})
        >>> table2 = Table({"b": [4, 5, 6]})
        >>> table1.add_tables_as_columns(table2)
        +-----+-----+
        |   a |   b |
        | --- | --- |
        | i64 | i64 |
        +===========+
        |   1 |   4 |
        |   2 |   5 |
        |   3 |   6 |
        +-----+-----+

        Related
        -------
        - [`add_tables_as_rows`][safeds.data.tabular.containers._table.Table.add_tables_as_rows]
        """
        import polars as pl

        if isinstance(others, Table):
            others = [others]

        _check_columns_dont_exist(self, [name for other in others for name in other.column_names])
        _check_row_counts_are_equal([self, *others], ignore_entries_without_rows=True)

        return Table._from_polars_lazy_frame(
            pl.concat(
                [
                    self._lazy_frame,
                    *[other._lazy_frame for other in others],
                ],
                how="horizontal",
            ),
        )

    def add_tables_as_rows(self, others: Table | list[Table]) -> Table:
        """
        Add the rows of other tables and return the result as a new table.

        **Note:** The original tables are not modified.

        Parameters
        ----------
        others:
            The tables to add as rows.

        Returns
        -------
        new_table:
            The table with the rows added.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table1 = Table({"a": [1, 2, 3]})
        >>> table2 = Table({"a": [4, 5, 6]})
        >>> table1.add_tables_as_rows(table2)
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

        Related
        -------
        - [`add_tables_as_columns`][safeds.data.tabular.containers._table.Table.add_tables_as_columns]
        """
        import polars as pl

        if isinstance(others, Table):
            others = [others]

        for other in others:
            _check_schema(self, other)

        return Table._from_polars_lazy_frame(
            pl.concat(
                [
                    self._lazy_frame,
                    *[other._lazy_frame for other in others],
                ],
                how="vertical",
            ),
        )

    def inverse_transform_table(self, fitted_transformer: InvertibleTableTransformer) -> Table:
        """
        Inverse-transform the table by a **fitted, invertible** transformer and return the result as a new table.

        **Notes:**

        - The original table is not modified.
        - Depending on the transformer, this operation might fully load the data into memory, which can be expensive.

        Parameters
        ----------
        fitted_transformer:
            The fitted, invertible transformer to apply.

        Returns
        -------
        new_table:
            The inverse-transformed table.

        Raises
        ------
        NotFittedError
            If the transformer has not been fitted yet.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> from safeds.data.tabular.transformation import RangeScaler
        >>> table = Table({"a": [1, 2, 3]})
        >>> transformer, transformed_table = RangeScaler(min=0, max=1).fit_and_transform(table)
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

        Related
        -------
        - [`transform_table`][safeds.data.tabular.containers._table.Table.transform_table]:
            Transform the table with a fitted transformer.
        """
        return fitted_transformer.inverse_transform(self)

    def join(
        self,
        right_table: Table,
        left_names: str | list[str],
        right_names: str | list[str],
        *,
        mode: Literal["inner", "left", "right", "full"] = "inner",
    ) -> Table:
        """
        Join the current table (left table) with another table (right table) and return the result as a new table.

        Rows are matched if the values in the specified columns are equal. The parameter `left_names` controls which
        columns are used for the left table, and `right_names` does the same for the right table.

        There are various types of joins, specified by the `mode` parameter:

        - `"inner"`:
            Keep only rows that have matching values in both tables.
        - `"left"`:
            Keep all rows from the left table and the matching rows from the right table. Cells with no match are
            marked as missing values.
        - `"right"`:
            Keep all rows from the right table and the matching rows from the left table. Cells with no match are
            marked as missing values.
        - `"full"`:
            Keep all rows from both tables. Cells with no match are marked as missing values.

        **Note:** The original tables are not modified.

        Parameters
        ----------
        right_table:
            The table to join with the left table.
        left_names:
            Names of columns to join on in the left table.
        right_names:
            Names of columns to join on in the right table.
        mode:
            Specify which type of join you want to use.

        Returns
        -------
        new_table:
            The table with the joined table.

        Raises
        ------
        ColumnNotFoundError
            If a column does not exist in one of the tables.
        DuplicateColumnError
            If a column is used multiple times in the join.
        LengthMismatchError
            If the number of columns to join on is different in the two tables.
        ValueError
            If `left_names` or `right_names` are an empty list.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table1 = Table({"a": [1, 2], "b": [True, False]})
        >>> table2 = Table({"c": [1, 3], "d": ["a", "b"]})
        >>> table1.join(table2, "a", "c", mode="inner")
        +-----+------+-----+
        |   a | b    | d   |
        | --- | ---  | --- |
        | i64 | bool | str |
        +==================+
        |   1 | true | a   |
        +-----+------+-----+

        >>> table1.join(table2, "a", "c", mode="left")
        +-----+-------+------+
        |   a | b     | d    |
        | --- | ---   | ---  |
        | i64 | bool  | str  |
        +====================+
        |   1 | true  | a    |
        |   2 | false | null |
        +-----+-------+------+

        >>> table1.join(table2, "a", "c", mode="right")
        +------+-----+-----+
        | b    |   c | d   |
        | ---  | --- | --- |
        | bool | i64 | str |
        +==================+
        | true |   1 | a   |
        | null |   3 | b   |
        +------+-----+-----+

        >>> table1.join(table2, "a", "c", mode="full")
        +-----+-------+------+
        |   a | b     | d    |
        | --- | ---   | ---  |
        | i64 | bool  | str  |
        +====================+
        |   1 | true  | a    |
        |   2 | false | null |
        |   3 | null  | b    |
        +-----+-------+------+
        """
        # Preprocessing
        if isinstance(left_names, str):
            left_names = [left_names]
        if isinstance(right_names, str):
            right_names = [right_names]

        # Validation
        _check_columns_exist(self, left_names)
        _check_columns_exist(right_table, right_names)

        duplicate_left_names = _compute_duplicates(left_names)
        if duplicate_left_names:
            raise DuplicateColumnError(
                f"Columns to join on must be unique, but left names {duplicate_left_names} are duplicated.",
            )

        duplicate_right_names = _compute_duplicates(right_names)
        if duplicate_right_names:
            raise DuplicateColumnError(
                f"Columns to join on must be unique, but right names {duplicate_right_names} are duplicated.",
            )

        if len(left_names) != len(right_names):
            raise LengthMismatchError("The number of columns to join on must be the same in both tables.")
        if not left_names or not right_names:
            # Here both are empty, due to the previous check
            raise ValueError("The columns to join on must not be empty.")

        # Implementation
        result = self._lazy_frame.join(
            right_table._lazy_frame,
            left_on=left_names,
            right_on=right_names,
            how=mode,
            maintain_order="left_right",
            coalesce=True,
        )

        return self._from_polars_lazy_frame(result)

    def transform_table(self, fitted_transformer: TableTransformer) -> Table:
        """
        Transform the table with a **fitted** transformer and return the result as a new table.

        **Notes:**

        - The original table is not modified.
        - Depending on the transformer, this operation might fully load the data into memory, which can be expensive.

        Parameters
        ----------
        fitted_transformer:
            The fitted transformer to apply.

        Returns
        -------
        new_table:
            The transformed table.

        Raises
        ------
        NotFittedError
            If the transformer has not been fitted yet.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> from safeds.data.tabular.transformation import RangeScaler
        >>> table = Table({"a": [1, 2, 3]})
        >>> transformer = RangeScaler(min=0, max=1).fit(table)
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

        Related
        -------
        - [`inverse_transform_table`][safeds.data.tabular.containers._table.Table.inverse_transform_table]:
            Inverse-transform the table with a fitted, invertible transformer.
        - [`transform_columns`][safeds.data.tabular.containers._table.Table.transform_columns]:
            Transform columns with a custom function.
        """
        return fitted_transformer.transform(self)

    # ------------------------------------------------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------------------------------------------------

    def summarize_statistics(self) -> Table:
        """
        Return a table with important statistics about this table.

        !!! warning "API Stability"

            Do not rely on the exact output of this method. In future versions, we may change the displayed statistics
            without prior notice.

        Returns
        -------
        statistics:
            The table with statistics.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 3]})
        >>> table.summarize_statistics()
        +---------------------+---------+
        | statistic           |       a |
        | ---                 |     --- |
        | str                 |     f64 |
        +===============================+
        | min                 | 1.00000 |
        | max                 | 3.00000 |
        | mean                | 2.00000 |
        | median              | 2.00000 |
        | standard deviation  | 1.41421 |
        | missing value ratio | 0.00000 |
        | stability           | 0.50000 |
        | idness              | 1.00000 |
        +---------------------+---------+
        """
        import polars as pl
        import polars.selectors as cs

        if self.column_count == 0:
            # polars raises an error in this case
            return Table({})

        # Find suitable name for the statistic column
        statistic_column_name = "statistic"
        while statistic_column_name in self.column_names:
            statistic_column_name += "_"

        # Build the expressions to compute the statistics
        non_null_columns = cs.exclude(cs.by_dtype(pl.Null))
        non_boolean_columns = cs.exclude(cs.by_dtype(pl.Boolean))
        boolean_columns = cs.by_dtype(pl.Boolean)
        true_count = boolean_columns.filter(boolean_columns == True).count()  # noqa: E712
        false_count = boolean_columns.filter(boolean_columns == False).count()  # noqa: E712

        named_statistics: dict[str, list[pl.Expr]] = {
            "min": [non_null_columns.min()],
            "max": [non_null_columns.max()],
            "mean": [cs.numeric().mean()],
            "median": [cs.numeric().median()],
            "standard deviation": [cs.numeric().std()],
            # NaN occurs for tables without rows
            "missing value ratio": [(cs.all().null_count() / pl.len()).fill_nan(1.0)],
            # null occurs for columns without non-null values
            # `unique_counts` crashes in polars for boolean columns (https://github.com/pola-rs/polars/issues/16356)
            "stability": [
                (non_boolean_columns.drop_nulls().unique_counts().max() / non_boolean_columns.count()).fill_null(1.0),
                (
                    pl.when(true_count >= false_count).then(true_count).otherwise(false_count) / boolean_columns.count()
                ).fill_null(1.0),
            ],
            # NaN occurs for tables without rows
            "idness": [(cs.all().n_unique() / pl.len()).fill_nan(1.0)],
        }

        # Compute suitable types for the output columns
        frame = self._lazy_frame
        schema = _safe_collect_lazy_frame_schema(frame)
        for name, type_ in schema.items():
            # polars fails to determine supertype of temporal types and u32
            if not type_.is_numeric() and not type_.is_(pl.Null):
                schema[name] = pl.String

        # Combine everything into a single table
        return Table._from_polars_lazy_frame(
            pl.concat(
                [
                    # Ensure the columns are in the correct order
                    pl.LazyFrame({statistic_column_name: []}),
                    schema.to_frame(eager=False),
                    # Add the statistics
                    *[
                        frame.select(
                            pl.lit(name).alias(statistic_column_name),
                            *expressions,
                        )
                        for name, expressions in named_statistics.items()
                    ],
                ],
                how="diagonal_relaxed",
            ),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------------------------------------------------------

    def to_columns(self) -> list[Column]:
        """
        Return the data of the table as a list of columns.

        Returns
        -------
        columns:
            The columns of the table.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> columns = table.to_columns()
        """
        return [Column._from_polars_lazy_frame(name, self._lazy_frame) for name in self.column_names]

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
        FileExtensionError
            If the path has an extension that is not ".csv".

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.to_csv_file("./src/resources/to_csv_file.csv")

        Related
        -------
        - [`to_json_file`][safeds.data.tabular.containers._table.Table.to_json_file]
        - [`to_parquet_file`][safeds.data.tabular.containers._table.Table.to_parquet_file]
        """
        path = _normalize_and_check_file_path(path, ".csv", [".csv"])
        path.parent.mkdir(parents=True, exist_ok=True)

        self._lazy_frame.sink_csv(path)

    def to_dict(self) -> dict[str, list[Any]]:
        """
        Return a dictionary that maps column names to column values.

        **Note:** This operation must fully load the data into memory, which can be expensive.

        Returns
        -------
        dict:
            The dictionary representation of the table.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.to_dict()
        {'a': [1, 2, 3], 'b': [4, 5, 6]}
        """
        return self._data_frame.to_dict(as_series=False)

    def to_json_file(
        self,
        path: str | Path,
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

        Raises
        ------
        FileExtensionError
            If the path has an extension that is not ".json".

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.to_json_file("./src/resources/to_json_file.json")

        Related
        -------
        - [`to_csv_file`][safeds.data.tabular.containers._table.Table.to_csv_file]
        - [`to_parquet_file`][safeds.data.tabular.containers._table.Table.to_parquet_file]
        """
        path = _normalize_and_check_file_path(path, ".json", [".json"])
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write JSON to file
        self._data_frame.write_json(path)

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
        FileExtensionError
            If the path has an extension that is not ".parquet".

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.to_parquet_file("./src/resources/to_parquet_file.parquet")

        Related
        -------
        - [`to_csv_file`][safeds.data.tabular.containers._table.Table.to_csv_file]
        - [`to_json_file`][safeds.data.tabular.containers._table.Table.to_json_file]
        """
        path = _normalize_and_check_file_path(path, ".parquet", [".parquet"])
        path.parent.mkdir(parents=True, exist_ok=True)

        self._lazy_frame.sink_parquet(path)

    def to_tabular_dataset(
        self,
        target_name: str,
        /,  # If we allow multiple targets in the future, we would rename the parameter to `target_names`.
        *,
        extra_names: str | list[str] | None = None,
    ) -> TabularDataset:
        """
        Return a new `TabularDataset` with columns marked as a target, feature, or extra.

        - The target column is the column that a model should predict.
        - Feature columns are columns that a model should use to make predictions.
        - Extra columns are columns that are neither feature nor target. They are ignored by models and can be used to
          provide additional context. An ID or name column is a common example.

        Feature columns are implicitly defined as all columns except the target and extra columns. If no extra columns
        are specified, all columns except the target column are used as features.

        Parameters
        ----------
        target_name:
            The name of the target column.
        extra_names:
            Names of the columns that are neither features nor target. If None, no extra columns are used, i.e. all but
            the target column are used as features.

        Raises
        ------
        ColumnNotFoundError
            If a target or extra column does not exist.
        ValueError
            If the target column is also an extra column.
        ValueError
            If no feature column remains.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table(
        ...     {
        ...         "extra": [1, 2, 3],
        ...         "feature": [4, 5, 6],
        ...         "target": [7, 8, 9],
        ...     },
        ... )
        >>> dataset = table.to_tabular_dataset("target", extra_names="extra")
        """
        from safeds.data.labeled.containers import TabularDataset  # circular import

        return TabularDataset(
            self,
            target_name,
            extra_names=extra_names,
        )

    # TODO: check design, test, and add the method back (here we should definitely allow multiple targets)
    # def to_time_series_dataset(
    #     self,
    #     target_name: str,
    #     window_size: int,
    #     *,
    #     extra_names: list[str] | None = None,
    #     forecast_horizon: int = 1,
    #     continuous: bool = False,
    # ) -> TimeSeriesDataset:
    #     """
    #     Return a new `TimeSeriesDataset` with columns marked as a target column, time or feature columns.
    #
    #     The original table is not modified.
    #
    #     Parameters
    #     ----------
    #     target_name:
    #         The name of the target column.
    #     window_size:
    #         The number of consecutive sample to use as input for prediction.
    #     extra_names:
    #         Names of the columns that are neither features nor target. If None, no extra columns are used, i.e. all but
    #         the target column are used as features.
    #     forecast_horizon:
    #         The number of time steps to predict into the future.
    #
    #     Returns
    #     -------
    #     dataset:
    #         A new time series dataset with the given target and feature names.
    #
    #     Raises
    #     ------
    #     ValueError
    #         If the target column is also a feature column.
    #     ValueError
    #         If the time column is also a feature column.
    #
    #     Examples
    #     --------
    #     >>> from safeds.data.tabular.containers import Table
    #     >>> table = Table({"day": [0, 1, 2], "price": [1.10, 1.19, 1.79], "amount_bought": [74, 72, 51]})
    #     >>> dataset = table.to_time_series_dataset("amount_bought", window_size=2)
    #     """
    #     from safeds.data.labeled.containers import TimeSeriesDataset  # circular import
    #
    #     return TimeSeriesDataset(
    #         self,
    #         target_name,
    #         window_size=window_size,
    #         extra_names=extra_names,
    #         forecast_horizon=forecast_horizon,
    #         continuous=continuous,
    #     )

    # ------------------------------------------------------------------------------------------------------------------
    # Dataframe interchange protocol
    # ------------------------------------------------------------------------------------------------------------------

    def __dataframe__(self, allow_copy: bool = True) -> DataFrame:
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

    # ------------------------------------------------------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------------------------------------------------------

    # TODO: check and potentially rework this
    def _into_dataloader(self, batch_size: int) -> DataLoader:
        """
        Return a Dataloader for the data stored in this table, used for predicting with neural networks.

        The original table is not modified.

        Parameters
        ----------
        batch_size:
            The size of data batches that should be loaded at one time.

        Returns
        -------
        result:
            The DataLoader.

        """
        import polars as pl
        import torch
        from torch.utils.data import DataLoader

        _init_default_device()

        return DataLoader(
            dataset=_create_dataset(self._data_frame.to_torch(dtype=pl.Float32).to(_get_device())),
            batch_size=batch_size,
            generator=torch.Generator(device=_get_device()),
        )


# TODO: check and potentially rework this
def _create_dataset(features: Tensor) -> Dataset:
    from torch.utils.data import Dataset

    _init_default_device()

    class _CustomDataset(Dataset):
        def __init__(self, features: Tensor):
            self.X = features
            self.len = self.X.shape[0]

        def __getitem__(self, item: int) -> torch.Tensor:
            return self.X[item]

        def __len__(self) -> int:
            return self.len

    return _CustomDataset(features)
