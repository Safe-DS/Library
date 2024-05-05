from __future__ import annotations

from typing import TYPE_CHECKING, Any

from safeds.exceptions import ColumnLengthMismatchError, IndexOutOfBoundsError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from polars import DataFrame, LazyFrame


class ExperimentalPolarsTable:
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
    # Creation
    # ------------------------------------------------------------------------------------------------------------------

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
