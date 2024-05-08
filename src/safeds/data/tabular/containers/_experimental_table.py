from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

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

    from safeds.data.tabular.transformation import InvertibleTableTransformer, TableTransformer
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
    ColumnLengthMismatchError
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
        import polars as pl

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
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 2   ┆ 1   │
        │ 0   ┆ 0   ┆ 7   │
        └─────┴─────┴─────┘
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
        ColumnLengthMismatchError
            If columns have different lengths.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        >>> ExperimentalTable.from_dict(data)
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 2   ┆ 5   │
        │ 3   ┆ 6   │
        └─────┴─────┘
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
        >>> ExperimentalTable.from_json_file("./src/resources/from_json_file.json")
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 2   ┆ 1   │
        │ 0   ┆ 0   ┆ 7   │
        └─────┴─────┴─────┘
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
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 2   ┆ 1   │
        │ 0   ┆ 0   ┆ 7   │
        └─────┴─────┴─────┘
        """
        import polars as pl

        path = _check_and_normalize_file_path(path, ".parquet", [".parquet"], check_if_file_exists=True)
        return ExperimentalTable._from_polars_lazy_frame(pl.scan_parquet(path))

    @staticmethod
    def _from_polars_data_frame(data: pl.DataFrame) -> ExperimentalTable:
        result = object.__new__(ExperimentalTable)
        result._lazy_frame = data.lazy()
        result._data_frame = data
        return result

    @staticmethod
    def _from_polars_lazy_frame(data: pl.LazyFrame) -> ExperimentalTable:
        result = object.__new__(ExperimentalTable)
        result._lazy_frame = data
        result._data_frame = None
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
        self._data_frame: pl.DataFrame | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExperimentalTable):
            return False
        if self is other:
            return True

        if self._data_frame is None:
            self._data_frame = self._lazy_frame.collect()
        if other._data_frame is None:
            other._data_frame = other._lazy_frame.collect()

        return self._data_frame.frame_equal(other._data_frame)

    def __hash__(self) -> int:
        return _structural_hash(self.schema, self.number_of_rows)

    def __repr__(self) -> str:
        if self._data_frame is None:
            self._data_frame = self._lazy_frame.collect()

        return self._data_frame.__repr__()

    def __sizeof__(self) -> int:
        if self._data_frame is None:
            self._data_frame = self._lazy_frame.collect()

        return self._data_frame.estimated_size()

    def __str__(self) -> str:
        if self._data_frame is None:
            self._data_frame = self._lazy_frame.collect()

        return self._data_frame.__str__()

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

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

        Note that this operation must fully load the data into memory, which can be expensive.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.number_of_rows
        3
        """
        if self._data_frame is None:
            self._data_frame = self._lazy_frame.collect()

        return self._data_frame.height

    @property
    def plot(self) -> ExperimentalTablePlotter:
        return ExperimentalTablePlotter(self)

    @property
    def schema(self) -> ExperimentalSchema:
        return _PolarsSchema(self._lazy_frame.schema)

    # ------------------------------------------------------------------------------------------------------------------
    # Column operations
    # ------------------------------------------------------------------------------------------------------------------

    def add_columns(
        self,
        columns: ExperimentalColumn | list[ExperimentalColumn],
    ) -> ExperimentalTable:
        if isinstance(columns, ExperimentalColumn):
            columns = [columns]

        if len(columns) == 0:
            return self

        if self._data_frame is None:
            self._data_frame = self._lazy_frame.collect()

        return ExperimentalTable._from_polars_data_frame(
            self._data_frame.hstack([column._series for column in columns]),
        )

    def compute_column(
        self,
        name: str,
        computer: Callable[[ExperimentalRow], ExperimentalCell],
    ) -> ExperimentalTable:
        if self.has_column(name):
            raise DuplicateColumnNameError(name)

        computed_column = computer(_LazyVectorizedRow(self))

        return self._from_polars_lazy_frame(
            self._lazy_frame.with_columns(name, computed_column._polars_expression),
        )

    def get_column(self, name: str) -> ExperimentalColumn:
        if self._data_frame is None:
            self._data_frame = self._lazy_frame.collect()

        return ExperimentalColumn._from_polars_series(self._data_frame.get_column(name))

    def get_column_type(self, name: str) -> ExperimentalDataType:
        return _PolarsDataType(self._lazy_frame.schema[name])

    def has_column(self, name: str) -> bool:
        return name in self.column_names

    def remove_columns_by_name(
        self,
        names: str | list[str],
        *,
        keep_only_listed: bool = False,
    ) -> ExperimentalTable:
        if isinstance(names, str):
            names = [names]

        if keep_only_listed:
            names_to_keep = set(names)  # perf: Comprehensions evaluate their condition every iteration
            names = [name for name in self.column_names if name not in names_to_keep]

        return ExperimentalTable._from_polars_lazy_frame(
            self._lazy_frame.drop(names),
        )

    def remove_columns_with_missing_values(self) -> ExperimentalTable:
        import polars as pl

        if self._data_frame is None:
            self._data_frame = self._lazy_frame.collect()

        return ExperimentalTable._from_polars_lazy_frame(
            pl.LazyFrame(
                [series for series in self._data_frame.get_columns() if series.null_count() == 0],
            ),
        )

    def remove_non_numeric_columns(self) -> ExperimentalTable:
        import polars.selectors as cs

        return ExperimentalTable._from_polars_lazy_frame(
            self._lazy_frame.select(cs.numeric()),
        )

    def rename_column(self, old_name: str, new_name: str) -> ExperimentalTable:
        """
        Return a new table with a column renamed.

        Note that the original table is not modified.

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

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.rename_column("a", "A")
        shape: (3, 2)
        ┌─────┬─────┐
        │ A   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 2   ┆ 5   │
        │ 3   ┆ 6   │
        └─────┴─────┘
        """
        # TODO: raises?
        return ExperimentalTable._from_polars_lazy_frame(
            self._lazy_frame.rename({old_name: new_name}),
        )

    def replace_column(
        self,
        old_name: str,
        new_columns: ExperimentalColumn | list[ExperimentalColumn],
    ) -> ExperimentalTable:
        if isinstance(new_columns, ExperimentalColumn):
            new_columns = [new_columns]

        if len(new_columns) == 0:
            return self.remove_columns_by_name(old_name)

        if self._data_frame is None:
            self._data_frame = self._lazy_frame.collect()

        new_frame = self._data_frame
        index = new_frame.get_column_index(old_name)

        if len(new_columns) == 1:
            return ExperimentalTable._from_polars_data_frame(
                new_frame.replace_column(index, new_columns[0]._series),
            )

        prefix = new_frame.select(self.column_names[:index])
        suffix = new_frame.select(self.column_names[index + 1:])

        return ExperimentalTable._from_polars_data_frame(
            prefix.hstack([column._series for column in new_columns]).hstack(suffix),
        )

    def transform_column(
        self,
        name: str,
        transformer: Callable[[ExperimentalCell], ExperimentalCell],
    ) -> ExperimentalTable:
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
        Remove duplicate rows from the table.

        Returns
        -------
        filtered_table:
            The table without duplicate rows.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, 2, 2], "b": [4, 5, 5]})
        >>> table.remove_duplicate_rows()
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 2   ┆ 5   │
        └─────┴─────┘
        """
        return ExperimentalTable._from_polars_lazy_frame(
            self._lazy_frame.unique(maintain_order=True),
        )

    def remove_rows(
        self,
        query: Callable[[ExperimentalRow], ExperimentalCell[bool]],
    ) -> ExperimentalTable:
        mask = query(_LazyVectorizedRow(self))

        return ExperimentalTable._from_polars_lazy_frame(
            self._lazy_frame.filter(mask._polars_expression),
        )

    def remove_rows_by_column(
        self,
        name: str,
        query: Callable[[ExperimentalCell], ExperimentalCell[bool]],
    ) -> ExperimentalTable:
        import polars as pl

        if not self.has_column(name):
            raise UnknownColumnNameError([name])

        mask = query(_LazyCell(pl.col(name)))

        return ExperimentalTable._from_polars_lazy_frame(
            self._lazy_frame.filter(mask._polars_expression),
        )

    def remove_rows_with_missing_values(
        self,
        subset_names: list[str] | None = None,
    ) -> ExperimentalTable:
        """
        Remove rows with missing values from the table.

        Note that the original table is not modified.

        Parameters
        ----------
        subset_names:
            Names of the columns to consider. If None, all columns are considered.

        Returns
        -------
        filtered_table:
            The table without rows containing missing values in the specified columns.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalTable
        >>> table = ExperimentalTable({"a": [1, None, 3], "b": [4, 5, None]})
        >>> table.remove_rows_with_missing_values()
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        └─────┴─────┘
        """
        return ExperimentalTable._from_polars_lazy_frame(
            self._lazy_frame.drop_nulls(subset=subset_names),
        )

    def remove_rows_with_outliers(
        self,
        subset_names: list[str] | None = None,
    ) -> ExperimentalTable:
        raise NotImplementedError

    def shuffle_rows(self) -> ExperimentalTable:
        if self._data_frame is None:
            self._data_frame = self._lazy_frame.collect()

        return ExperimentalTable._from_polars_data_frame(
            self._data_frame.sample(
                fraction=1,
                shuffle=True,
                seed=_get_random_seed(),
            ),
        )

    def slice_rows(self, start: int = 0, length: int | None = None) -> ExperimentalTable:
        return ExperimentalTable._from_polars_lazy_frame(
            self._lazy_frame.slice(start, length),
        )

    def sort_rows(
        self,
        key_selector: Callable[[ExperimentalRow], ExperimentalCell],
        *,
        descending: bool = False,
    ) -> ExperimentalTable:
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
        if self._data_frame is None:
            self._data_frame = self._lazy_frame.collect()

        return ExperimentalTable._from_polars_data_frame(
            self._data_frame.hstack(other._data_frame),
        )

    def add_table_as_rows(self, other: ExperimentalTable) -> ExperimentalTable:
        if self._data_frame is None:
            self._data_frame = self._lazy_frame.collect()

        return ExperimentalTable._from_polars_data_frame(
            self._data_frame.vstack(other._data_frame),
        )

    def inverse_transform_table(self, fitted_transformer: InvertibleTableTransformer) -> ExperimentalTable:
        # TODO: more efficient implementation
        # old_table = self.temporary_to_old_table().inverse_transform_table(fitted_transformer)
        # return ExperimentalTable._from_polars_data_frame(
        #     pl.DataFrame(old_table.)
        # )
        raise NotImplementedError

    def transform_table(self, fitted_transformer: TableTransformer) -> ExperimentalTable:
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------------------------------------------------

    def summarize_statistics(self) -> ExperimentalTable:
        if self.number_of_columns == 0:
            return ExperimentalTable()

        head = self.get_column(self.column_names[0]).summarize_statistics()
        tail = [
            self.get_column(name).summarize_statistics().get_column(name)._series
            for name in self.column_names[1:]
        ]

        return ExperimentalTable._from_polars_data_frame(
            head._lazy_frame.collect().hstack(tail, in_place=True),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------------------------------------------------------

    def to_columns(self) -> list[ExperimentalColumn]:
        if self._data_frame is None:
            self._data_frame = self._lazy_frame.collect()

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
        if self._data_frame is None:
            self._data_frame = self._lazy_frame.collect()

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

        Note that this operation must fully load the data into memory, which can be expensive.

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
        >>> table.to_json_file("./src/resources/to_json_file.json")
        """
        path = _check_and_normalize_file_path(path, ".json", [".json"])
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write JSON to file
        if self._data_frame is None:
            self._data_frame = self._lazy_frame.collect()

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
        if self._data_frame is None:
            self._data_frame = self._lazy_frame.collect()

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

        Note that this operation must fully load the data into memory, which can be expensive.

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
        if self._data_frame is None:
            self._data_frame = self._lazy_frame.collect()

        return self._data_frame.__dataframe__(allow_copy=allow_copy)

    # ------------------------------------------------------------------------------------------------------------------
    # IPython integration
    # ------------------------------------------------------------------------------------------------------------------

    def _repr_html_(self) -> str:
        """
        Return a compact HTML representation of the table for IPython.

        Note that this operation must fully load the data into memory, which can be expensive.

        Returns
        -------
        html:
            The generated HTML.
        """
        if self._data_frame is None:
            self._data_frame = self._lazy_frame.collect()

        return self._data_frame._repr_html_()
