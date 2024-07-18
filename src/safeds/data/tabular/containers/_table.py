from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

from safeds._config import _get_device, _init_default_device
from safeds._config._polars import _get_polars_config
from safeds._utils import _structural_hash
from safeds._utils._random import _get_random_seed
from safeds._validation import _check_bounds, _check_columns_exist, _ClosedBound, _normalize_and_check_file_path
from safeds._validation._check_columns_dont_exist import _check_columns_dont_exist
from safeds.data.tabular.plotting import TablePlotter
from safeds.data.tabular.typing._polars_data_type import _PolarsDataType
from safeds.data.tabular.typing._polars_schema import _PolarsSchema
from safeds.exceptions import (
    ColumnLengthMismatchError,
    DuplicateColumnError,
)

from ._column import Column
from ._lazy_cell import _LazyCell
from ._lazy_vectorized_row import _LazyVectorizedRow

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    import polars as pl
    import torch
    from torch import Tensor
    from torch.utils.data import DataLoader, Dataset

    from safeds.data.labeled.containers import TabularDataset, TimeSeriesDataset
    from safeds.data.tabular.transformation import (
        InvertibleTableTransformer,
        TableTransformer,
    )
    from safeds.data.tabular.typing import DataType, Schema

    from ._cell import Cell
    from ._row import Row


class Table:
    """
    A two-dimensional collection of data. It can either be seen as a list of rows or as a list of columns.

    To create a `Table` call the constructor or use one of the following static methods:

    | Method                                                                             | Description                            |
    | ---------------------------------------------------------------------------------- | -------------------------------------- |
    | [from_csv_file][safeds.data.tabular.containers._table.Table.from_csv_file]         | Create a table from a CSV file.        |
    | [from_json_file][safeds.data.tabular.containers._table.Table.from_json_file]       | Create a table from a JSON file.       |
    | [from_parquet_file][safeds.data.tabular.containers._table.Table.from_parquet_file] | Create a table from a Parquet file.    |
    | [from_columns][safeds.data.tabular.containers._table.Table.from_columns]           | Create a table from a list of columns. |
    | [from_dict][safeds.data.tabular.containers._table.Table.from_dict]                 | Create a table from a dictionary.      |

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
    >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def from_columns(columns: Column | list[Column]) -> Table:
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
        from polars.exceptions import DuplicateError, ShapeError

        if isinstance(columns, Column):
            columns = [columns]

        try:
            return Table._from_polars_lazy_frame(
                pl.LazyFrame([column._series for column in columns]),
            )
        except DuplicateError:
            raise DuplicateColumnError("") from None  # TODO: message
        except ShapeError:
            raise ColumnLengthMismatchError("") from None  # TODO: message

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
        FileNotFoundError
            If no file exists at the given path.
        ValueError
            If the path has an extension that is not ".csv".

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
        ValueError
            If columns have different lengths.

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
        FileNotFoundError
            If no file exists at the given path.
        ValueError
            If the path has an extension that is not ".json".

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
        """
        import polars as pl

        path = _normalize_and_check_file_path(path, ".json", [".json"], check_if_file_exists=True)

        try:
            return Table._from_polars_data_frame(pl.read_json(path))
        except pl.PolarsPanicError:
            # Can happen if the JSON file is empty (https://github.com/pola-rs/polars/issues/10234)
            return Table()

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
        FileNotFoundError
            If no file exists at the given path.
        ValueError
            If the path has an extension that is not ".parquet".

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
        import polars as pl

        if self.__data_frame_cache is None:
            try:
                self.__data_frame_cache = self._lazy_frame.collect()
            except (pl.NoDataError, pl.PolarsPanicError):
                # Can happen for some operations on empty tables (e.g. https://github.com/pola-rs/polars/issues/16202)
                return pl.DataFrame()

        return self.__data_frame_cache

    @property
    def column_names(self) -> list[str]:
        """
        The names of the columns in the table.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.column_names
        ['a', 'b']
        """
        return self.schema.column_names

    @property
    def column_count(self) -> int:
        """
        The number of columns in the table.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.column_count
        2
        """
        import polars as pl

        try:
            return self._lazy_frame.width
        except (pl.NoDataError, pl.PolarsPanicError):
            # Can happen for some operations on empty tables (e.g. https://github.com/pola-rs/polars/issues/16202)
            return 0

    @property
    def row_count(self) -> int:
        """
        The number of rows in the table.

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
        """The plotter for the table."""
        return TablePlotter(self)

    @property
    def schema(self) -> Schema:
        """The schema of the table."""
        import polars as pl

        try:
            return _PolarsSchema(self._lazy_frame.schema)
        except (pl.NoDataError, pl.PolarsPanicError):
            # Can happen for some operations on empty tables (e.g. https://github.com/pola-rs/polars/issues/16202)
            return _PolarsSchema({})

    # ------------------------------------------------------------------------------------------------------------------
    # Column operations
    # ------------------------------------------------------------------------------------------------------------------

    def add_columns(
        self,
        columns: Column | list[Column],
    ) -> Table:
        """
        Return a new table with additional columns.

        **Notes:**

        - The original table is not modified.
        - This operation must fully load the data into memory, which can be expensive.

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
        ValueError
            If a column name already exists.
        ValueError
            If the columns have incompatible lengths.

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
        """
        import polars as pl

        if isinstance(columns, Column):
            columns = [columns]

        if len(columns) == 0:
            return self

        try:
            return Table._from_polars_data_frame(
                self._data_frame.hstack([column._series for column in columns]),
            )
        except pl.DuplicateError:
            # polars already validates this, so we don't need to do it again upfront (performance)
            _check_columns_dont_exist(self, [column.name for column in columns])
            return Table()  # pragma: no cover

    def add_computed_column(
        self,
        name: str,
        computer: Callable[[Row], Cell],
    ) -> Table:
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
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
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
            raise DuplicateColumnError(name)

        computed_column = computer(_LazyVectorizedRow(self))

        return self._from_polars_lazy_frame(
            self._lazy_frame.with_columns(computed_column._polars_expression.alias(name)),
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
        return Column._from_polars_series(
            self._lazy_frame.select(name).collect().get_column(name),
        )

    def get_column_type(self, name: str) -> DataType:
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
        ColumnNotFoundError
            If the column does not exist.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.get_column_type("a")
        Int64
        """
        _check_columns_exist(self, name)
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
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.has_column("a")
        True
        """
        return name in self.column_names

    def remove_columns(
        self,
        names: str | list[str],
        /,
        *,
        ignore_unknown_names: bool = False,
    ) -> Table:
        """
        Return a new table without the specified columns.

        **Notes:**

        - The original table is not modified.

        Parameters
        ----------
        names:
            The names of the columns to remove.
        ignore_unknown_names:
            If set to True, columns that are not present in the table will be ignored.
            If set to False, an error will be raised if any of the specified columns do not exist.

        Returns
        -------
        new_table:
            The table with the columns removed.

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
        """
        if isinstance(names, str):
            names = [names]

        if not ignore_unknown_names:
            _check_columns_exist(self, names)

        return Table._from_polars_lazy_frame(
            self._lazy_frame.drop(names),
        )

    def remove_columns_except(
        self,
        names: str | list[str],
        /,
    ) -> Table:
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
        ColumnNotFoundError
            If a column does not exist.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
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

        _check_columns_exist(self, names)

        return Table._from_polars_lazy_frame(
            self._lazy_frame.select(names),
        )

    def remove_columns_with_missing_values(self) -> Table:
        """
        Return a new table without columns that contain missing values.

        **Notes:**

        - The original table is not modified.
        - This operation must fully load the data into memory, which can be expensive.

        Returns
        -------
        new_table:
            The table without columns containing missing values.

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
        """
        import polars as pl

        return Table._from_polars_lazy_frame(
            pl.LazyFrame(
                [series for series in self._data_frame.get_columns() if series.null_count() == 0],
            ),
        )

    def remove_non_numeric_columns(self) -> Table:
        """
        Return a new table without non-numeric columns.

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
        """
        import polars.selectors as cs

        return Table._from_polars_lazy_frame(
            self._lazy_frame.select(cs.numeric()),
        )

    def rename_column(self, old_name: str, new_name: str) -> Table:
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
        Return a new table with a column replaced by zero or more columns.

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
        if isinstance(new_columns, Column):
            new_columns = [new_columns]
        elif isinstance(new_columns, Table):
            new_columns = new_columns.to_columns()

        _check_columns_exist(self, old_name)
        _check_columns_dont_exist(self, [column.name for column in new_columns], old_name=old_name)

        if len(new_columns) == 0:
            return self.remove_columns(old_name, ignore_unknown_names=True)

        if len(new_columns) == 1:
            new_column = new_columns[0]
            return Table._from_polars_lazy_frame(
                self._lazy_frame.with_columns(new_column._series.alias(old_name)).rename({old_name: new_column.name}),
            )

        import polars as pl

        index = self.column_names.index(old_name)

        return Table._from_polars_lazy_frame(
            self._lazy_frame.select(
                *[pl.col(name) for name in self.column_names[:index]],
                *[column._series for column in new_columns],
                *[pl.col(name) for name in self.column_names[index + 1 :]],
            ),
        )

    def transform_column(
        self,
        name: str,
        transformer: Callable[[Cell], Cell],
    ) -> Table:
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
        ColumnNotFoundError
            If no column with the specified name exists.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
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
        _check_columns_exist(self, name)

        import polars as pl

        expression = transformer(_LazyCell(pl.col(name)))

        return Table._from_polars_lazy_frame(
            self._lazy_frame.with_columns(expression._polars_expression),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Row operations
    # ------------------------------------------------------------------------------------------------------------------

    @overload
    def count_row_if(
        self,
        predicate: Callable[[Row], Cell[bool | None]],
        *,
        ignore_unknown: Literal[True] = ...,
    ) -> int: ...

    @overload
    def count_row_if(
        self,
        predicate: Callable[[Row], Cell[bool | None]],
        *,
        ignore_unknown: bool,
    ) -> int | None: ...

    def count_row_if(
        self,
        predicate: Callable[[Row], Cell[bool | None]],
        *,
        ignore_unknown: bool = True,
    ) -> int | None:
        """
        Return how many rows in the table satisfy the predicate.

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
        >>> table = Table({"col1": [1, 2, 3], "col2": [1, 3, 3]})
        >>> table.count_row_if(lambda row: row["col1"] == row["col2"])
        2

        >>> table.count_row_if(lambda row: row["col1"] > row["col2"])
        0
        """
        expression = predicate(_LazyVectorizedRow(self))._polars_expression
        series = self._lazy_frame.select(expression.alias("count")).collect().get_column("count")

        if ignore_unknown or series.null_count() == 0:
            return series.sum()
        else:
            return None

    # TODO: Rethink group_rows/group_rows_by_column. They should not return a dict.

    def remove_duplicate_rows(self) -> Table:
        """
        Return a new table without duplicate rows.

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
        """
        return Table._from_polars_lazy_frame(
            self._lazy_frame.unique(maintain_order=True),
        )

    def remove_rows(
        self,
        query: Callable[[Row], Cell[bool]],
    ) -> Table:
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
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
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

        return Table._from_polars_lazy_frame(
            self._lazy_frame.filter(~mask._polars_expression),
        )

    def remove_rows_by_column(
        self,
        name: str,
        query: Callable[[Cell], Cell[bool]],
    ) -> Table:
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
        """
        _check_columns_exist(self, name)

        import polars as pl

        mask = query(_LazyCell(pl.col(name)))

        return Table._from_polars_lazy_frame(
            self._lazy_frame.filter(~mask._polars_expression),
        )

    def remove_rows_with_missing_values(
        self,
        column_names: list[str] | None = None,
    ) -> Table:
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
        """
        return Table._from_polars_lazy_frame(
            self._lazy_frame.drop_nulls(subset=column_names),
        )

    def remove_rows_with_outliers(
        self,
        column_names: list[str] | None = None,
        *,
        z_score_threshold: float = 3,
    ) -> Table:
        """
        Return a new table without rows containing outliers in the specified columns.

        Whether a data point is an outlier in a column is determined by its z-score. The z-score the distance of the
        data point from the mean of the column divided by the standard deviation of the column. If the z-score is
        greater than the given threshold, the data point is considered an outlier. Missing values are ignored during the
        calculation of the z-score.

        The z-score is only defined for numeric columns. Non-numeric columns are ignored, even if they are specified in
        `column_names`.

        **Notes:**

        - The original table is not modified.
        - This operation must fully load the data into memory, which can be expensive.

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
        """
        if self.row_count == 0:
            return self  # polars raises a ComputeError for tables without rows
        if column_names is None:
            column_names = self.column_names

        import polars as pl
        import polars.selectors as cs

        non_outlier_mask = pl.all_horizontal(
            self._data_frame.select(cs.numeric() & cs.by_name(column_names)).select(
                pl.all().is_null() | (((pl.all() - pl.all().mean()) / pl.all().std()).abs() <= z_score_threshold),
            ),
        )

        return Table._from_polars_lazy_frame(
            self._lazy_frame.filter(non_outlier_mask),
        )

    def shuffle_rows(self) -> Table:
        """
        Return a new table with the rows shuffled.

        **Note:** The original table is not modified.

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
        |   3 |   6 |
        |   2 |   5 |
        |   1 |   4 |
        +-----+-----+
        """
        return Table._from_polars_data_frame(
            self._data_frame.sample(
                fraction=1,
                shuffle=True,
                seed=_get_random_seed(),
            ),
        )

    def slice_rows(self, start: int = 0, length: int | None = None) -> Table:
        """
        Return a new table with a slice of rows.

        **Note:** The original table is not modified.

        Parameters
        ----------
        start:
            The start index of the slice.
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
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [2, 1, 3], "b": [1, 1, 2]})
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
        if self.row_count == 0:
            return self

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
    ) -> tuple[Table, Table]:
        """
        Create two tables by splitting the rows of the current table.

        The first table contains a percentage of the rows specified by `percentage_in_first`, and the second table
        contains the remaining rows.

        **Notes:**

        - The original table is not modified.
        - By default, the rows are shuffled before splitting. You can disable this by setting `shuffle` to False.

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
        _check_bounds(
            "percentage_in_first",
            percentage_in_first,
            lower_bound=_ClosedBound(0),
            upper_bound=_ClosedBound(1),
        )

        input_table = self.shuffle_rows() if shuffle else self
        row_count_in_first = round(percentage_in_first * input_table.row_count)

        return (
            input_table.slice_rows(length=row_count_in_first),
            input_table.slice_rows(start=row_count_in_first),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Table operations
    # ------------------------------------------------------------------------------------------------------------------

    def add_table_as_columns(self, other: Table) -> Table:
        """
        Return a new table with the columns of another table added.

        **Notes:**

        - The original tables are not modified.
        - This operation must fully load the data into memory, which can be expensive.

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
        >>> from safeds.data.tabular.containers import Table
        >>> table1 = Table({"a": [1, 2, 3]})
        >>> table2 = Table({"b": [4, 5, 6]})
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

        return Table._from_polars_data_frame(
            self._data_frame.hstack(other._data_frame),
        )

    def add_table_as_rows(self, other: Table) -> Table:
        """
        Return a new table with the rows of another table added.

        **Notes:**

        - The original tables are not modified.
        - This operation must fully load the data into memory, which can be expensive.

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
        >>> from safeds.data.tabular.containers import Table
        >>> table1 = Table({"a": [1, 2, 3]})
        >>> table2 = Table({"a": [4, 5, 6]})
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

        return Table._from_polars_data_frame(
            self._data_frame.vstack(other._data_frame),
        )

    def inverse_transform_table(self, fitted_transformer: InvertibleTableTransformer) -> Table:
        """
        Return a new table inverse-transformed by a **fitted, invertible** transformer.

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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> from safeds.data.tabular.transformation import RangeScaler
        >>> table = Table({"a": [1, 2, 3]})
        >>> transformer, transformed_table = RangeScaler(min_=0, max_=1, column_names="a").fit_and_transform(table)
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

    def join(
        self,
        right_table: Table,
        left_names: str | list[str],
        right_names: str | list[str],
        *,
        mode: Literal["inner", "left", "outer"] = "inner",
    ) -> Table:
        """
        Join a table with the current table and return the result.

        Parameters
        ----------
        right_table:
            The other table which is to be joined to the current table.
        left_names:
            Name or list of names of columns from the current table on which to join right_table.
        right_names:
            Name or list of names of columns from right_table on which to join the current table.
        mode:
            Specify which type of join you want to use. Options include 'inner', 'outer', 'left', 'right'.

        Returns
        -------
        new_table:
            The table with the joined table.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table1 = Table({"a": [1, 2], "b": [3, 4]})
        >>> table2 = Table({"d": [1, 5], "e": [5, 6]})
        >>> table1.join(table2, "a", "d", mode="left")
        +-----+-----+------+
        |   a |   b |    e |
        | --- | --- |  --- |
        | i64 | i64 |  i64 |
        +==================+
        |   1 |   3 |    5 |
        |   2 |   4 | null |
        +-----+-----+------+
        """
        # Validation
        _check_columns_exist(self, left_names)
        _check_columns_exist(right_table, right_names)

        if len(left_names) != len(right_names):
            raise ValueError("The number of columns to join on must be the same in both tables.")

        # Implementation
        return self._from_polars_lazy_frame(
            self._lazy_frame.join(
                right_table._lazy_frame,
                left_on=left_names,
                right_on=right_names,
                how=mode,
            ),
        )

    def transform_table(self, fitted_transformer: TableTransformer) -> Table:
        """
        Return a new table transformed by a **fitted** transformer.

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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> from safeds.data.tabular.transformation import RangeScaler
        >>> table = Table({"a": [1, 2, 3]})
        >>> transformer = RangeScaler(min_=0, max_=1, column_names="a").fit(table)
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

    def summarize_statistics(self) -> Table:
        """
        Return a table with important statistics about this table.

        Returns
        -------
        statistics:
            The table with statistics.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 3]})
        >>> table.summarize_statistics()
        +----------------------+---------+
        | metric               |       a |
        | ---                  |     --- |
        | str                  |     f64 |
        +================================+
        | min                  | 1.00000 |
        | max                  | 3.00000 |
        | mean                 | 2.00000 |
        | median               | 2.00000 |
        | standard deviation   | 1.41421 |
        | distinct value count | 2.00000 |
        | idness               | 1.00000 |
        | missing value ratio  | 0.00000 |
        | stability            | 0.50000 |
        +----------------------+---------+
        """
        if self.column_count == 0:
            return Table()

        head = self.get_column(self.column_names[0]).summarize_statistics()
        tail = [self.get_column(name).summarize_statistics().get_column(name)._series for name in self.column_names[1:]]

        return Table._from_polars_data_frame(
            head._lazy_frame.collect().hstack(tail, in_place=True),
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
            List of columns.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> columns = table.to_columns()
        """
        return [Column._from_polars_series(column) for column in self._data_frame.get_columns()]

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
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.to_csv_file("./src/resources/to_csv_file.csv")
        """
        path = _normalize_and_check_file_path(path, ".csv", [".csv"])
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
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
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
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.to_json_file("./src/resources/to_json_file.json")
        """
        path = _normalize_and_check_file_path(path, ".json", [".json"])
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
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.to_parquet_file("./src/resources/to_parquet_file.parquet")
        """
        path = _normalize_and_check_file_path(path, ".parquet", [".parquet"])
        path.parent.mkdir(parents=True, exist_ok=True)

        self._lazy_frame.sink_parquet(path)

    def to_tabular_dataset(self, target_name: str, *, extra_names: list[str] | None = None) -> TabularDataset:
        """
        Return a new `TabularDataset` with columns marked as a target, feature, or extra.

        - The target column is the column that a model should predict.
        - Feature columns are columns that a model should use to make predictions.
        - Extra columns are columns that are neither feature nor target. They can be used to provide additional context,
          like an ID column.

        Feature columns are implicitly defined as all columns except the target and extra columns. If no extra columns
        are specified, all columns except the target column are used as features.

        Parameters
        ----------
        target_name:
            The name of the target column.
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
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table(
        ...     {
        ...         "item": ["apple", "milk", "beer"],
        ...         "price": [1.10, 1.19, 1.79],
        ...         "amount_bought": [74, 72, 51],
        ...     }
        ... )
        >>> dataset = table.to_tabular_dataset(target_name="amount_bought", extra_names=["item"])
        """
        from safeds.data.labeled.containers import TabularDataset  # circular import

        return TabularDataset(
            self,
            target_name=target_name,
            extra_names=extra_names,
        )

    def to_time_series_dataset(
        self,
        target_name: str,
        window_size: int,
        *,
        extra_names: list[str] | None = None,
        forecast_horizon: int = 1,
        continuous: bool = False,
    ) -> TimeSeriesDataset:
        """
        Return a new `TimeSeriesDataset` with columns marked as a target column, time or feature columns.

        The original table is not modified.

        Parameters
        ----------
        target_name:
            The name of the target column.
        window_size:
            The number of consecutive sample to use as input for prediction.
        extra_names:
            Names of the columns that are neither features nor target. If None, no extra columns are used, i.e. all but
            the target column are used as features.
        forecast_horizon:
            The number of time steps to predict into the future.

        Returns
        -------
        dataset:
            A new time series dataset with the given target and feature names.

        Raises
        ------
        ValueError
            If the target column is also a feature column.
        ValueError
            If the time column is also a feature column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Table({"day": [0, 1, 2], "price": [1.10, 1.19, 1.79], "amount_bought": [74, 72, 51]})
        >>> dataset = table.to_time_series_dataset(target_name="amount_bought", window_size=2)
        """
        from safeds.data.labeled.containers import TimeSeriesDataset  # circular import

        return TimeSeriesDataset(
            self,
            target_name=target_name,
            window_size=window_size,
            extra_names=extra_names,
            forecast_horizon=forecast_horizon,
            continuous=continuous,
        )

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

    # ------------------------------------------------------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------------------------------------------------------

    # TODO
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


# TODO
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
