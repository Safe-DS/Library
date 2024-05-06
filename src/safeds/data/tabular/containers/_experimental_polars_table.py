from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from safeds.data.tabular.containers import Table
from safeds.exceptions import ColumnLengthMismatchError, WrongFileExtensionError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from polars import DataFrame, LazyFrame

    from safeds.data.labeled.containers import TabularDataset
    from safeds.data.tabular.containers._experimental_polars_column import ExperimentalPolarsColumn


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
        WrongFileExtensionError
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

        # Handle file extension
        path = Path(path)
        if path.suffix == "":
            path = path.with_suffix(".csv")
        elif path.suffix != ".csv":
            raise WrongFileExtensionError(path, ".csv")

        # Read CSV file
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

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def number_of_rows(self) -> int:
        """
        The number of rows in the table.

        Returns
        -------
        number_of_rows:
            The number of rows.

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

    # ------------------------------------------------------------------------------------------------------------------
    # Row operations
    # ------------------------------------------------------------------------------------------------------------------

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

    def remove_rows_with_missing_values(self) -> ExperimentalPolarsTable:
        """
        Remove rows with missing values from the table.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalPolarsTable
        >>> table = ExperimentalPolarsTable({"a": [1, None, 3], "b": [4, 5, None]})
        >>> table.remove_rows_with_missing_values()
           a  b
        0  1  4
        """
        return ExperimentalPolarsTable._from_polars_lazy_frame(self._lazy_frame.drop_nulls())

    # def slice_rows(self, start: int = 0, size: int | None = None) -> ExperimentalPolarsTable:
    #     """
    #     Slice the rows of the table.
    #
    #     Parameters
    #     ----------
    #     start:
    #         The start index.
    #     size:
    #         The size of the slice. If None, all rows from the start index are included.
    #
    #     Returns
    #     -------
    #     table:
    #         The sliced table.
    #
    #     Examples
    #     --------
    #     >>> from safeds.data.tabular.containers import ExperimentalPolarsTable
    #     >>> table = ExperimentalPolarsTable({"a": [1, 2, 3], "b": [4, 5, 6]})
    #     >>> table.slice_rows(start=1, end=3)
    #        a  b
    #     0  2  5
    #     1  3  6
    #     """
    #
    #
    #
    #     if end is None:
    #         end = self.number_of_rows
    #
    #     if end < start:
    #         raise IndexOutOfBoundsError(slice(start, end))
    #     if start < 0 or end < 0 or start > self.number_of_rows or end > self.number_of_rows:
    #         raise IndexOutOfBoundsError(start if start < 0 or start > self.number_of_rows else end)
    #
    #     return self._lazy_frame.slice(start, end)

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
        WrongFileExtensionError
            If the path has an extension that is not ".csv".

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalPolarsTable
        >>> table = ExperimentalPolarsTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.to_csv_file("./src/resources/to_csv_file.csv")
        """
        # Handle file extension
        path = Path(path)
        if path.suffix == "":
            path = path.with_suffix(".csv")
        elif path.suffix != ".csv":
            raise WrongFileExtensionError(path, ".csv")

        # Ensure that parent directories exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write CSV to file
        if self._data_frame is None:
            self._data_frame = self._lazy_frame.collect()

        self._data_frame.write_csv(path)

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

    def to_json_file(self, path: str | Path) -> None:
        """
        Write the table to a JSON file.

        If the file and/or the parent directories do not exist, they will be created. If the file exists already, it
        will be overwritten.

        Parameters
        ----------
        path:
            The path to the JSON file. If the file extension is omitted, it is assumed to be ".json".

        Raises
        ------
        WrongFileExtensionError
            If the path has an extension that is not ".json".

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalPolarsTable
        >>> table = ExperimentalPolarsTable({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> table.to_json_file("./src/resources/to_json_file.json")
        """
        # Handle file extension
        path = Path(path)
        if path.suffix == "":
            path = path.with_suffix(".json")
        elif path.suffix != ".json":
            raise WrongFileExtensionError(path, ".json")

        # Ensure that parent directories exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write JSON to file
        if self._data_frame is None:
            self._data_frame = self._lazy_frame.collect()

        self._data_frame.write_json(path)

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
