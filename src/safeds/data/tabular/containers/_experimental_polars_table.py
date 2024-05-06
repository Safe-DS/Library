from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

from safeds._utils import _check_and_normalize_file_path
from safeds.data.tabular.containers import Table
from safeds.exceptions import ColumnLengthMismatchError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from polars import DataFrame, LazyFrame

    from safeds.data.image.containers import Image
    from safeds.data.labeled.containers import TabularDataset
    from safeds.data.tabular.transformation import InvertibleTableTransformer, TableTransformer
    from safeds.data.tabular.typing import ColumnType, Schema

    from ._experimental_polars_cell import ExperimentalPolarsCell
    from ._experimental_polars_column import ExperimentalPolarsColumn
    from ._experimental_polars_row import ExperimentalPolarsRow

_T: TypeVar = TypeVar("_T")


class ExperimentalPolarsTable:
    """
    A table is a two-dimensional collection of data. It can either be seen as a list of rows or as a list of columns.

    To create a `Table` call the constructor or use one of the following static methods:

    | Method                                                                                                             | Description                            |
    | ------------------------------------------------------------------------------------------------------------------ | -------------------------------------- |
    | [from_csv_file][safeds.data.tabular.containers._experimental_polars_table.ExperimentalPolarsTable.from_csv_file]   | Create a table from a CSV file.        |
    | [from_json_file][safeds.data.tabular.containers._experimental_polars_table.ExperimentalPolarsTable.from_json_file] | Create a table from a JSON file.       |
    | [from_dict][safeds.data.tabular.containers._experimental_polars_table.ExperimentalPolarsTable.from_dict]           | Create a table from a dictionary.      |
    | [from_columns][safeds.data.tabular.containers._experimental_polars_table.ExperimentalPolarsTable.from_columns]     | Create a table from a list of columns. |
    | [from_rows][safeds.data.tabular.containers._experimental_polars_table.ExperimentalPolarsTable.from_rows]           | Create a table from a list of rows.    |

    Parameters
    ----------
    data:
        The data. If None, an empty table is created.

    Raises
    ------
    ColumnLengthMismatchError
        If columns have different lengths.

    Examples
    --------
    >>> from safeds.data.tabular.containers import ExperimentalPolarsTable
    >>> table = ExperimentalPolarsTable({"a": [1, 2, 3], "b": [4, 5, 6]})
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def from_columns(columns: list[ExperimentalPolarsColumn]) -> ExperimentalPolarsTable:
        raise NotImplementedError

    @staticmethod
    def from_csv_file(path: str | Path) -> ExperimentalPolarsTable:
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
        >>> from safeds.data.tabular.containers import ExperimentalPolarsTable
        >>> table = ExperimentalPolarsTable.from_csv_file("./src/resources/from_csv_file.csv")
           a  b  c
        0  1  2  1
        1  0  0  7
        """
        import polars as pl

        path = _check_and_normalize_file_path(path, ".csv", [".csv"], check_if_file_exists=True)
        return ExperimentalPolarsTable._from_polars_lazy_frame(pl.scan_csv(path))

    @staticmethod
    def from_dict(data: dict[str, list[Any]]) -> ExperimentalPolarsTable:
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
        >>> from safeds.data.tabular.containers import ExperimentalPolarsTable
        >>> d = {'a': [1, 2], 'b': [3, 4]}
        >>> ExperimentalPolarsTable.from_dict(d)
           a  b
        0  1  3
        1  2  4
        """
        return ExperimentalPolarsTable(data)

    @staticmethod
    def from_json_file(path: str | Path) -> ExperimentalPolarsTable:
        raise NotImplementedError

    @staticmethod
    def from_parquet_file(path: str | Path) -> ExperimentalPolarsTable:
        raise NotImplementedError

    @staticmethod
    def _from_polars_dataframe(data: DataFrame) -> ExperimentalPolarsTable:
        result = object.__new__(ExperimentalPolarsTable)
        result._lazy_frame = data.lazy()
        result._data_frame = data
        return result

    @staticmethod
    def _from_polars_lazy_frame(data: LazyFrame) -> ExperimentalPolarsTable:
        result = object.__new__(ExperimentalPolarsTable)
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
        raise NotImplementedError

    def __hash__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    def __sizeof__(self) -> int:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def column_names(self) -> list[str]:
        """
        Names of the columns in the table.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalPolarsTable
        >>> table = ExperimentalPolarsTable({"a": [1, 2, 3], "b": [4, 5, 6]})
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
        >>> from safeds.data.tabular.containers import ExperimentalPolarsTable
        >>> table = ExperimentalPolarsTable({"a": [1, 2, 3], "b": [4, 5, 6]})
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
        >>> from safeds.data.tabular.containers import ExperimentalPolarsTable
        >>> table = ExperimentalPolarsTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.number_of_rows
        3
        """
        if self._data_frame is None:
            self._data_frame = self._lazy_frame.collect()

        return self._data_frame.height

    @property
    def schema(self) -> Schema:  # TODO: rethink return type
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------
    # Column operations
    # ------------------------------------------------------------------------------------------------------------------

    def add_columns(
        self,
        columns: ExperimentalPolarsColumn | list[ExperimentalPolarsColumn],
    ) -> ExperimentalPolarsTable:
        raise NotImplementedError

    def get_column(self, name: str) -> ExperimentalPolarsColumn:
        raise NotImplementedError

    def get_column_type(self, name: str) -> ColumnType:  # TODO rethink return type
        raise NotImplementedError

    def has_column(self, name: str) -> bool:
        raise NotImplementedError

    def remove_columns(self, names: list[str]) -> ExperimentalPolarsTable:
        raise NotImplementedError

    def remove_columns_except(self, names: list[str]) -> ExperimentalPolarsTable:
        raise NotImplementedError

    def remove_columns_with_missing_values(self) -> ExperimentalPolarsTable:
        raise NotImplementedError

    def remove_columns_with_non_numerical_values(self) -> ExperimentalPolarsTable:
        raise NotImplementedError

    def rename_column(self, old_name: str, new_name: str) -> ExperimentalPolarsTable:
        """
        Return a new table with a column renamed.

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
        >>> from safeds.data.tabular.containers import ExperimentalPolarsTable
        >>> table = ExperimentalPolarsTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.rename_column("a", "A")
        """
        # TODO: raises?
        return ExperimentalPolarsTable._from_polars_lazy_frame(
            self._lazy_frame.rename({old_name: new_name}),
        )

    def replace_column(
        self,
        old_name: str,
        new_columns: ExperimentalPolarsColumn | list[ExperimentalPolarsColumn],
    ) -> ExperimentalPolarsTable:
        raise NotImplementedError

    def transform_column(
        self,
        name: str,
        transformer: Callable[[ExperimentalPolarsRow], ExperimentalPolarsCell],
    ) -> ExperimentalPolarsTable:
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------
    # Row operations
    # ------------------------------------------------------------------------------------------------------------------

    # TODO: Rethink group_rows/group_rows_by_column. They should not return a dict.

    def remove_duplicate_rows(self) -> ExperimentalPolarsTable:
        """
        Remove duplicate rows from the table.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalPolarsTable
        >>> table = ExperimentalPolarsTable({"a": [1, 2, 2], "b": [4, 5, 5]})
        >>> table.remove_duplicate_rows()
           a  b
        0  1  4
        1  2  5
        """
        return ExperimentalPolarsTable._from_polars_lazy_frame(self._lazy_frame.unique())

    def remove_rows(
        self,
        query: Callable[[ExperimentalPolarsRow], ExperimentalPolarsCell[bool]],
    ) -> ExperimentalPolarsTable:
        raise NotImplementedError

    def remove_rows_by_column(
        self,
        name: str,
        query: Callable[[ExperimentalPolarsCell], ExperimentalPolarsCell[bool]],
    ) -> ExperimentalPolarsTable:
        raise NotImplementedError

    def remove_rows_with_missing_values(
        self,
        column_names: list[str] | None = None,
    ) -> ExperimentalPolarsTable:
        """
        Remove rows with missing values from the table.

        Parameters
        ----------
        column_names:
            Names of the columns to consider. If None, all columns are considered.

        Returns
        -------
        filtered_table:
            The table without rows containing missing values in the specified columns.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalPolarsTable
        >>> table = ExperimentalPolarsTable({"a": [1, None, 3], "b": [4, 5, None]})
        >>> table.remove_rows_with_missing_values()
           a  b
        0  1  4
        """
        return ExperimentalPolarsTable._from_polars_lazy_frame(
            self._lazy_frame.drop_nulls(subset=column_names),
        )

    def remove_rows_with_outliers(
        self,
        column_names: list[str] | None = None,
    ):
        raise NotImplementedError

    def shuffle_rows(self) -> ExperimentalPolarsTable:
        raise NotImplementedError

    def slice_rows(self, start: int = 0, end: int | None = None):
        raise NotImplementedError

    def sort_rows(
        self,
        key_selector: Callable[[ExperimentalPolarsRow], ExperimentalPolarsCell],
        *,
        descending: bool = False,
    ) -> ExperimentalPolarsTable:
        raise NotImplementedError

    def sort_rows_by_column(
        self,
        name: str,
        *,
        descending: bool = False,
        maintain_order_if_equal: bool = False,
    ) -> ExperimentalPolarsTable:
        raise NotImplementedError

    def split_rows(
        self,
        percentage_in_first: float,
        *,
        shuffle: bool = True,
    ) -> tuple[ExperimentalPolarsTable, ExperimentalPolarsTable]:
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------
    # Table operations
    # ------------------------------------------------------------------------------------------------------------------

    def add_table_as_columns(self, other: ExperimentalPolarsTable) -> ExperimentalPolarsTable:
        raise NotImplementedError

    def add_table_as_rows(self, other: ExperimentalPolarsTable) -> ExperimentalPolarsTable:
        raise NotImplementedError

    def inverse_transform_table(self, fitted_transformer: InvertibleTableTransformer) -> ExperimentalPolarsTable:
        raise NotImplementedError

    def transform_table(self, fitted_transformer: TableTransformer) -> ExperimentalPolarsTable:
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------------------------------------------------

    def summarize_statistics(self) -> ExperimentalPolarsTable:
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------------------------------------------------------

    def plot_boxplots(self) -> Image:
        raise NotImplementedError

    def plot_correlation_heatmap(self) -> Image:
        raise NotImplementedError

    def plot_histograms(self, *, number_of_bins: int = 10) -> Image:
        raise NotImplementedError

    def plot_lineplot(self, x_name: str, y_name: str) -> Image:
        raise NotImplementedError

    def plot_scatterplot(self, x_name: str, y_name: str) -> Image:
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------------------------------------------------------

    def to_columns(self) -> list[ExperimentalPolarsColumn]:
        raise NotImplementedError

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
        >>> from safeds.data.tabular.containers import ExperimentalPolarsTable
        >>> table = ExperimentalPolarsTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.to_csv_file("./src/resources/to_csv_file.csv")
        """
        path = _check_and_normalize_file_path(path, ".csv", [".csv"])
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._data_frame is None:
            self._data_frame = self._lazy_frame.collect()

        self._data_frame.write_csv(path)  # TODO: replace with LazyFrame.sink_csv once stable

    def to_dict(self) -> dict[str, list[Any]]:
        """
        Return a dictionary that maps column names to column values.

        Returns
        -------
        dict_:
            Dictionary representation of the table.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalPolarsTable
        >>> table = ExperimentalPolarsTable({"a": [1, 2, 3], "b": [4, 5, 6]})
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
        >>> from safeds.data.tabular.containers import ExperimentalPolarsTable
        >>> table = ExperimentalPolarsTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.to_json_file("./src/resources/to_json_file.json")
        """
        path = _check_and_normalize_file_path(path, ".json", [".json"])
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write JSON to file
        if self._data_frame is None:
            self._data_frame = self._lazy_frame.collect()

        self._data_frame.write_json(path, row_oriented=(orientation == "row"))

    def to_parquet_file(self, path: str | Path) -> None:
        raise NotImplementedError

    def to_tabular_dataset(self, target_name: str, extra_names: list[str] | None = None) -> TabularDataset:
        """
        Return a new `TabularDataset` with columns marked as a target, feature, or extra.

        * The target column is the column that a model should predict.
        * Feature columns are columns that a model should use to make predictions.
        * Extra columns are columns that are neither feature nor target. They can be used to provide additional context,
          like an ID or name column.

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
        >>> from safeds.data.tabular.containers import ExperimentalPolarsTable
        >>> table = ExperimentalPolarsTable({"item": ["apple", "milk", "beer"], "price": [1.10, 1.19, 1.79], "amount_bought": [74, 72, 51]})
        >>> dataset = table.to_tabular_dataset(target_name="amount_bought", extra_names=["item"])
        """
        from safeds.data.labeled.containers import TabularDataset

        # TODO: more efficient implementation
        return TabularDataset(self.temporary_to_old_table(), target_name, extra_names)

    def temporary_to_old_table(self) -> Table:
        """
        Convert the table to the old table format. This method is temporary and will be removed in a later version.

        Returns
        -------
        old_table:
            The table in the old format.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalPolarsTable
        >>> table = ExperimentalPolarsTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.temporary_to_old_table()
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

        Returns
        -------
        html:
            The generated HTML.
        """
        if self._data_frame is None:
            self._data_frame = self._lazy_frame.collect()

        return self._data_frame._repr_html_()
