from __future__ import annotations

import io
import sys
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xxhash

from safeds.data.image.containers import Image
from safeds.data.tabular.containers import Column, Row, Table, TaggedTable
from safeds.exceptions import (
    ColumnIsTargetError,
    ColumnIsTimeError,
    IllegalSchemaModificationError,
    NonNumericColumnError,
    UnknownColumnNameError,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Any


class TimeSeries(Table):

    # ------------------------------------------------------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _from_tagged_table(
        tagged_table: TaggedTable,
        time_name: str,
    ) -> TimeSeries:
        """Create a time series from a tagged table.

        Parameters
        ----------
        tagged_table: TaggedTable
            The tagged table.
        time_name: str
            Name of the time column.

        Returns
        -------
        time_series : TimeSeries
            the created time series

        Raises
        ------
        UnknownColumnNameError
            If time_name matches none of the column names.
        Value Error
            If time column is also a feature column

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table, TimeSeries
        >>> tagged_table = TaggedTable({"date": ["01.01", "01.02", "01.03", "01.04"], "col1": ["a", "b", "c", "a"]}, "col1" )
        >>> timeseries = TimeSeries._from_tagged_table(tagged_table, time_name = "date")
        """
        if time_name not in tagged_table.column_names:
            raise UnknownColumnNameError([time_name])
        table = tagged_table._as_table()
        # make sure that the time_name is not part of the features
        result = object.__new__(TimeSeries)
        feature_names = tagged_table.features.column_names
        if time_name in feature_names:
            feature_names.remove(time_name)

        if time_name == tagged_table.target.name:
            raise ValueError(f"Column '{time_name}' cannot be both time column and target.")

        result._data = table._data
        result._schema = table.schema
        result._time = table.get_column(time_name)
        result._features = table.keep_only_columns(feature_names)
        result._target = table.get_column(tagged_table.target.name)
        return result

    @staticmethod
    def _from_table(
        table: Table,
        target_name: str,
        time_name: str,
        feature_names: list[str] | None = None,
    ) -> TimeSeries:
        """Create a TimeSeries from a table.

        Parameters
        ----------
        table : Table
            The table.
        target_name : str
            Name of the target column.
        time_name: str
            Name of the date column.
        feature_names : list[str] | None
            Names of the feature columns. If None, all columns except the target and time columns are used.

        Returns
        -------
        time_series : TimeSeries
            the created time series

        Raises
        ------
        UnknownColumnNameError
            If target_name or time_name matches none of the column names.
        Value Error
            If one column is target and feature
        Value Error
            If one column is time and feature

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table, TimeSeries
        >>> test_table = Table({"date": ["01.01", "01.02", "01.03", "01.04"], "f1": ["a", "b", "c", "a"], "t": [1,2,3,4]})
        >>> timeseries = TimeSeries._from_table(test_table, "t", "date", ["f1"])
        """
        table = table._as_table()
        if feature_names is not None and time_name in feature_names:
            raise ValueError(f"Column '{time_name}' can not be time and feature column.")
        if feature_names is not None and target_name in feature_names:
            raise ValueError(f"Column '{target_name}' can not be target and feature column.")

        if target_name not in table.column_names:
            raise UnknownColumnNameError([target_name])

        result = object.__new__(TimeSeries)
        result._data = table._data
        result._schema = table._schema
        result._time = table.get_column(time_name)
        result._target = table.get_column(target_name)
        if feature_names is None or len(feature_names) == 0:
            result._feature_names = []
            result._features = Table()
        else:
            result._feature_names = feature_names
            result._features = table.keep_only_columns(feature_names)

        # check if time column got added as feature column
        return result

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        data: Mapping[str, Sequence[Any]],
        target_name: str,
        time_name: str,
        feature_names: list[str] | None = None,
    ):
        """
        Create a time series from a mapping of column names to their values.

        Parameters
        ----------
        data : Mapping[str, Sequence[Any]]
            The data.
        target_name : str
            Name of the target column.
        time_name : str
            Name of the time column
        feature_names : list[str] | None
            Names of the feature columns. If None, all columns except the target and time columns are used.

        Raises
        ------
        ColumnLengthMismatchError
            If columns have different lengths.
        ValueError
            If the target column is also a feature column.
        ValueError
            If time column is also a feature column
        UnknownColumnNameError
            If time column does not exist

        Examples
        --------
        >>> from safeds.data.tabular.containers import TaggedTable
        >>> table = TaggedTable({"a": [1, 2, 3], "b": [4, 5, 6]}, "b", ["a"])
        """
        # Validate inputs
        super().__init__(data)
        _data = Table(data)
        if feature_names is None:
            self._features = Table()
            self._feature_names = []
            feature_names = []
        else:
            self._feature_names = feature_names
            self._features = _data.keep_only_columns(feature_names)
        if time_name in feature_names:
            raise ValueError(f"Column '{time_name}' can not be time and feature column.")
        if target_name in feature_names:
            raise ValueError(f"Column '{target_name}' can not be time and feature column.")
        if time_name not in _data.column_names:
            raise UnknownColumnNameError([time_name])
        self._time: Column = _data.get_column(time_name)
        self._target: Column = _data.get_column(target_name)

    def __eq__(self, other: object) -> bool:
        """
        Compare two time series instances.

        Returns
        -------
        'True' if contents are equal, 'False' otherwise.
        """
        if not isinstance(other, TimeSeries):
            return NotImplemented
        if self is other:
            return True
        return (
            self.time == other.time
            and self.target == other.target
            and self.features == other.features
            and Table.__eq__(self, other)
        )

    def __hash__(self) -> int:
        """
        Return a deterministic hash value for this time series.

        Returns
        -------
        hash : int
            The hash value.
        """
        return xxhash.xxh3_64(
            hash(self.time).to_bytes(8)
            + hash(self.target).to_bytes(8)
            + hash(self.features).to_bytes(8)
            + Table.__hash__(self).to_bytes(8),
        ).intdigest()

    def __sizeof__(self) -> int:
        """
        Return the complete size of this object.

        Returns
        -------
        Size of this object in bytes.
        """
        return Table.__sizeof__(self) + sys.getsizeof(self._time)

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def target(self) -> Column:
        """
        Get the target column of the tagged table.

        Returns
        -------
        Column
            The target column.
        """
        return self._target

    @property
    def features(self) -> Table:
        """
        Get the feature columns of the tagged table.

        Returns
        -------
        Table
            The table containing the feature columns.
        """
        return self._features

    @property
    def time(self) -> Column:
        """
        Get the time column of the time series.

        Returns
        -------
        Column
            The time column.
        """
        return self._time

    # ------------------------------------------------------------------------------------------------------------------
    # Overriden methods from TaggedTable class:
    # ------------------------------------------------------------------------------------------------------------------
    def _as_table(self: TimeSeries) -> Table:
        """
        Return a new `Table` with the tagging removed.

        The original time series is not modified.

        Parameters
        ----------
        self: TimeSeries
            The Time Series.

        Returns
        -------
        table: Table
            The time series as an untagged Table, i.e. without the information about which columns are features, target or time.

        """
        return Table.from_columns(super().to_columns())

    def add_column(self, column: Column) -> TimeSeries:
        """
        Return a new `TimeSeries` with the provided column attached at the end, as neither target nor feature column.

        The original time series is not modified.

        Parameters
        ----------
        column : Column
            The column to be added.

        Returns
        -------
        result : TimeSeries
            The time series with the column attached as neither target nor feature column.

        Raises
        ------
        DuplicateColumnNameError
            If the new column already exists.
        ColumnSizeError
            If the size of the column does not match the number of rows.
        """
        return TimeSeries._from_table(
            super().add_column(column),
            time_name=self.time.name,
            target_name=self._target.name,
        )

    def add_column_as_feature(self, column: Column) -> TimeSeries:
        """
        Return a new `TimeSeries` with the provided column attached at the end, as a feature column.

        the original time series is not modified.

        Parameters
        ----------
        column : Column
            The column to be added.

        Returns
        -------
        result : TimeSeries
            The time series with the attached feature column.

        Raises
        ------
        DuplicateColumnNameError
            If the new column already exists.
        ColumnSizeError
            If the size of the column does not match the number of rows.
        """
        return TimeSeries._from_table(
            super().add_column(column),
            target_name=self._target.name,
            time_name=self.time.name,
            feature_names=[*self._feature_names, column.name],
        )

    def add_columns_as_features(self, columns: list[Column] | Table) -> TimeSeries:
        """
        Return a new `TimeSeries` with the provided columns attached at the end, as feature columns.

        The original time series is not modified.

        Parameters
        ----------
        columns : list[Column] | Table
            The columns to be added as features.

        Returns
        -------
        result : TimeSeries
            The time series with the attached feature columns.

        Raises
        ------
        DuplicateColumnNameError
            If any of the new feature columns already exist.
        ColumnSizeError
            If the size of any feature column does not match the number of rows.
        """
        return TimeSeries._from_table(
            super().add_columns(columns),
            time_name=self.time.name,
            target_name=self._target.name,
            feature_names=self._feature_names
            + [col.name for col in (columns.to_columns() if isinstance(columns, Table) else columns)],
        )

    def add_columns(self, columns: list[Column] | Table) -> TimeSeries:
        """
        Return a new `TimeSeries` with multiple added columns, as neither target nor feature columns.

        The original time series is not modified.

        Parameters
        ----------
        columns : list[Column] or Table
            The columns to be added.

        Returns
        -------
        result: TimeSeries
            A new time series combining the original table and the given columns as neither target nor feature columns.

        Raises
        ------
        DuplicateColumnNameError
            If at least one column name from the provided column list already exists in the time series.
        ColumnSizeError
            If at least one of the column sizes from the provided column list does not match the time series.
        """
        return TimeSeries._from_table(
            super().add_columns(columns),
            time_name=self.time.name,
            target_name=self._target.name,
            feature_names=self._feature_names,
        )

    def add_row(self, row: Row) -> TimeSeries:
        """
        Return a new `TimeSeries` with an extra Row attached.

        The original time series is not modified.

        Parameters
        ----------
        row : Row
            The row to be added.

        Returns
        -------
        table : TimeSeries
            A new time series with the added row at the end.

        Raises
        ------
        UnknownColumnNameError
            If the row has different column names than the time series.
        """
        return TimeSeries._from_table(
            super().add_row(row),
            target_name=self._target.name,
            time_name=self.time.name,
            feature_names=self._feature_names,
        )

    def add_rows(self, rows: list[Row] | Table) -> TimeSeries:
        """
        Return a new `TimeSeries` with multiple extra Rows attached.

        The original time series is not modified.

        Parameters
        ----------
        rows : list[Row] or Table
            The rows to be added.

        Returns
        -------
        result : TimeSeries
            A new time series which combines the original time series and the given rows.

        Raises
        ------
        UnknownColumnNameError
            If at least one of the rows have different column names than the time series.
        """
        return TimeSeries._from_table(
            super().add_rows(rows),
            target_name=self._target.name,
            time_name=self.time.name,
            feature_names=self._feature_names,
        )

    def filter_rows(self, query: Callable[[Row], bool]) -> TimeSeries:
        """
        Return a new `TimeSeries` containing only rows that match the given Callable (e.g. lambda function).

        The original time series is not modified.

        Parameters
        ----------
        query : lambda function
            A Callable that is applied to all rows.

        Returns
        -------
        result: TimeSeries
            A time series containing only the rows to match the query.
        """
        return TimeSeries._from_table(
            super().filter_rows(query),
            target_name=self._target.name,
            time_name=self.time.name,
            feature_names=self._feature_names,
        )

    def keep_only_columns(self, column_names: list[str]) -> TimeSeries:
        """
        Return a new `TimeSeries` with only the given column(s).

        The original time series is not modified.

        Parameters
        ----------
        column_names : list[str]
            A list containing the columns to be kept.

        Returns
        -------
        table : TimeSeries
            A time series containing only the given column(s).

        Raises
        ------
        UnknownColumnNameError
            If any of the given columns does not exist.
        IllegalSchemaModificationError
            If none of the given columns is the target or time column or any of the feature columns.
        """
        if self._target.name not in column_names:
            raise IllegalSchemaModificationError("Must keep the target column.")
        if self.time.name not in column_names:
            raise IllegalSchemaModificationError("Must keep the time column.")
        return TimeSeries._from_table(
            super().keep_only_columns(column_names),
            target_name=self._target.name,
            time_name=self.time.name,
            feature_names=sorted(
                set(self._feature_names).intersection(set(column_names)),
                key={val: ix for ix, val in enumerate(self._feature_names)}.__getitem__,
            ),
        )

    def remove_columns(self, column_names: list[str]) -> TimeSeries:
        """
        Return a new `TimeSeries` with the given column(s) removed from the time series.

        The original time series is not modified.

        Parameters
        ----------
        column_names : list[str]
            The names of all columns to be dropped.

        Returns
        -------
        table : TimeSeries
            A time series without the given columns.

        Raises
        ------
        UnknownColumnNameError
            If any of the given columns does not exist.
        ColumnIsTargetError
            If any of the given columns is the target column.
        ColumnIsTimeError
            If any of the given columns is the time column.
        IllegalSchemaModificationError
            If the given columns contain all the feature columns.
        """
        if self._target.name in column_names:
            raise ColumnIsTargetError(self._target.name)
        if self.time.name in column_names:
            raise ColumnIsTimeError(self.time.name)
        return TimeSeries._from_table(
            super().remove_columns(column_names),
            target_name=self._target.name,
            time_name=self.time.name,
            feature_names=sorted(
                set(self._feature_names) - set(column_names),
                key={val: ix for ix, val in enumerate(self._feature_names)}.__getitem__,
            ),
        )

    def remove_columns_with_missing_values(self) -> TimeSeries:
        """
        Return a new `TimeSeries` with every column that misses values removed.

        The original time series is not modified.

        Returns
        -------
        table : TimeSeries
            A time series without the columns that contain missing values.

        Raises
        ------
        ColumnIsTargetError
            If any of the columns to be removed is the target column.
        ColumnIsTimeError
            If any of the columns to be removed is the time column.
        IllegalSchemaModificationError
            If the columns to remove contain all the feature columns.
        """
        table = super().remove_columns_with_missing_values()
        if self._target.name not in table.column_names:
            raise ColumnIsTargetError(self._target.name)
        if self.time.name not in table.column_names:
            raise ColumnIsTimeError(self.time.name)
        return TimeSeries._from_table(
            table,
            target_name=self._target.name,
            time_name=self._time.name,
            feature_names=sorted(
                set(self._feature_names).intersection(set(table.column_names)),
                key={val: ix for ix, val in enumerate(self._feature_names)}.__getitem__,
            ),
        )

    def remove_columns_with_non_numerical_values(self) -> TimeSeries:
        """
        Return a new `TimeSeries` with every column that contains non-numerical values removed.

        The original time series is not modified.

        Returns
        -------
        table : TimeSeries
            A time series without the columns that contain non-numerical values.

        Raises
        ------
        ColumnIsTargetError
            If any of the columns to be removed is the target column.
        ColumnIsTimeError
            If any of the columns to be removed is the time column.
        IllegalSchemaModificationError
            If the columns to remove contain all the feature columns.
        """
        table = super().remove_columns_with_non_numerical_values()
        if self._target.name not in table.column_names:
            raise ColumnIsTargetError(self._target.name)
        if self.time.name not in table.column_names:
            raise ColumnIsTimeError(self.time.name)
        return TimeSeries._from_table(
            table,
            self._target.name,
            time_name=self.time.name,
            feature_names=sorted(
                set(self._feature_names).intersection(set(table.column_names)),
                key={val: ix for ix, val in enumerate(self._feature_names)}.__getitem__,
            ),
        )

    def remove_duplicate_rows(self) -> TimeSeries:
        """
        Return a new `TimeSeries` with all row duplicates removed.

        The original time series is not modified.

        Returns
        -------
        result : TimeSeries
            The time series with the duplicate rows removed.
        """
        return TimeSeries._from_table(
            super().remove_duplicate_rows(),
            target_name=self._target.name,
            feature_names=self._feature_names,
            time_name=self.time.name,
        )

    def remove_rows_with_missing_values(self) -> TimeSeries:
        """
        Return a new `TimeSeries` without the rows that contain missing values.

        The original time series is not modified.

        Returns
        -------
        table : TimeSeries
            A time series without the rows that contain missing values.
        """
        return TimeSeries._from_table(
            super().remove_rows_with_missing_values(),
            time_name=self.time.name,
            target_name=self._target.name,
            feature_names=self._feature_names,
        )

    def remove_rows_with_outliers(self) -> TimeSeries:
        """
        Return a new `TimeSeries` with all rows that contain at least one outlier removed.

        We define an outlier as a value that has a distance of more than 3 standard deviations from the column mean.
        Missing values are not considered outliers. They are also ignored during the calculation of the standard
        deviation.

        The original time series is not modified.

        Returns
        -------
        new_time_series : TimeSeries
            A new time series without rows containing outliers.
        """
        return TimeSeries._from_table(
            super().remove_rows_with_outliers(),
            time_name=self.time.name,
            target_name=self._target.name,
            feature_names=self._feature_names,
        )

    def rename_column(self, old_name: str, new_name: str) -> TimeSeries:
        """
        Return a new `TimeSeries` with a single column renamed.

        The original time series is not modified.

        Parameters
        ----------
        old_name : str
            The old name of the column.
        new_name : str
            The new name of the column.

        Returns
        -------
        table : TimeSeries
            The time series with the renamed column.

        Raises
        ------
        UnknownColumnNameError
            If the specified old target column name does not exist.
        DuplicateColumnNameError
            If the specified new target column name already exists.
        """
        return TimeSeries._from_table(
            super().rename_column(old_name, new_name),
            time_name=new_name if self.time.name == old_name else self.time.name,
            target_name=new_name if self._target.name == old_name else self._target.name,
            feature_names=(
                self._feature_names
                if old_name not in self._feature_names
                else [column_name if column_name != old_name else new_name for column_name in self._feature_names]
            ),
        )

    def replace_column(self, old_column_name: str, new_columns: list[Column]) -> TimeSeries:
        """
        Return a new `TimeSeries` with the specified old column replaced by a list of new columns.

        If the column to be replaced is the target or time column, it must be replaced by exactly one column. That column
        becomes the new target or time column. If the column to be replaced is a feature column, the new columns that replace it
        all become feature columns.

        The order of columns is kept. The original time series is not modified.

        Parameters
        ----------
        old_column_name : str
            The name of the column to be replaced.
        new_columns : list[Column]
            The new columns replacing the old column.

        Returns
        -------
        result : TimeSeries
            A time series with the old column replaced by the new columns.

        Raises
        ------
        UnknownColumnNameError
            If the old column does not exist.
        DuplicateColumnNameError
            If the new column already exists and the existing column is not affected by the replacement.
        ColumnSizeError
            If the size of the column does not match the amount of rows.
        IllegalSchemaModificationError
            If the target or time column would be removed or replaced by more than one column.
        """
        if old_column_name == self.time.name:
            if len(new_columns) != 1:
                raise IllegalSchemaModificationError(
                    f'Time column "{self.time.name}" can only be replaced by exactly one new column.',
                )
            else:
                return TimeSeries._from_table(
                    super().replace_column(old_column_name, new_columns),
                    target_name=self._target.name,
                    feature_names=self._feature_names,
                    time_name=new_columns[0].name,
                )
        if old_column_name == self._target.name:
            if len(new_columns) != 1:
                raise IllegalSchemaModificationError(
                    f'Target column "{self._target.name}" can only be replaced by exactly one new column.',
                )
            else:
                return TimeSeries._from_table(
                    super().replace_column(old_column_name, new_columns),
                    target_name=new_columns[0].name,
                    time_name=self.time.name,
                    feature_names=self._feature_names,
                )

        else:
            return TimeSeries._from_table(
                super().replace_column(old_column_name, new_columns),
                target_name=self._target.name,
                time_name=self.time.name,
                feature_names=(
                    self._feature_names
                    if old_column_name not in self._feature_names
                    else self._feature_names[: self._feature_names.index(old_column_name)]
                    + [col.name for col in new_columns]
                    + self._feature_names[self._feature_names.index(old_column_name) + 1 :]
                ),
            )

    def slice_rows(
        self,
        start: int | None = None,
        end: int | None = None,
        step: int = 1,
    ) -> TimeSeries:
        """
        Slice a part of the table into a new `TimeSeries`.

        The original time series is not modified.

        Parameters
        ----------
        start : int | None
            The first index of the range to be copied into a new time series, None by default.
        end : int | None
            The last index of the range to be copied into a new time series, None by default.
        step : int
            The step size used to iterate through the time series, 1 by default.

        Returns
        -------
        result : TimeSeries
            The resulting time series.

        Raises
        ------
        IndexOutOfBoundsError
            If the index is out of bounds.
        """
        return TimeSeries._from_table(
            super().slice_rows(start, end, step),
            target_name=self._target.name,
            feature_names=self._feature_names,
            time_name=self.time.name,
        )

    def sort_columns(
        self,
        comparator: Callable[[Column, Column], int] = lambda col1, col2: (col1.name > col2.name)
        - (col1.name < col2.name),
    ) -> TimeSeries:
        """
        Sort the columns of a `TimeSeries` with the given comparator and return a new `TimeSeries`.

        The comparator is a function that takes two columns `col1` and `col2` and
        returns an integer:

        * If the function returns a negative number, `col1` will be ordered before `col2`.
        * If the function returns a positive number, `col1` will be ordered after `col2`.
        * If the function returns 0, the original order of `col1` and `col2` will be kept.

        If no comparator is given, the columns will be sorted alphabetically by their name.

        The original time series is not modified.

        Parameters
        ----------
        comparator : Callable[[Column, Column], int]
            The function used to compare two columns.

        Returns
        -------
        new_time_series : TimeSeries
            A new time series with sorted columns.
        """
        sorted_table = super().sort_columns(comparator)
        return TimeSeries._from_table(
            sorted_table,
            time_name=self.time.name,
            target_name=self._target.name,
            feature_names=sorted(
                set(sorted_table.column_names).intersection(self._feature_names),
                key={val: ix for ix, val in enumerate(sorted_table.column_names)}.__getitem__,
            ),
        )

    def transform_column(self, name: str, transformer: Callable[[Row], Any]) -> TimeSeries:
        """
        Return a new `TimeSeries` with the provided column transformed by calling the provided transformer.

        The original time series is not modified.

        Parameters
        ----------
        name:
            The name of the column to be transformed.
        transformer:
            The transformer to the given column

        Returns
        -------
        result : TimeSeries
            The time series with the transformed column.

        Raises
        ------
        UnknownColumnNameError
            If the column does not exist.
        """
        return TimeSeries._from_table(
            super().transform_column(name, transformer),
            time_name=self.time.name,
            target_name=self._target.name,
            feature_names=self._feature_names,
        )

    def plot_lagplot(self, lag: int) -> Image:
        """
        Plot a lagplot for the target column.

        Parameters
        ----------
        lag:
            The amount of lag used to plot

        Returns
        -------
        plot:
            The plot as an image.

        Raises
        ------
        NonNumericColumnError
            If the time series targets contains non-numerical values.

        Examples
        --------
                >>> from safeds.data.tabular.containers import TimeSeries
                >>> table = TimeSeries({"time":[1, 2], "target": [3, 4], "feature":[2,2]}, target_name= "target", time_name="time", feature_names=["feature"], )
                >>> image = table.plot_lagplot(lag = 1)

        """
        if not self._target.type.is_numeric():
            raise NonNumericColumnError("This time series target contains non-numerical columns.")
        ax = pd.plotting.lag_plot(self._target._data, lag=lag)
        fig = ax.figure
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        plt.close()  # Prevents the figure from being displayed directly
        buffer.seek(0)
        return Image.from_bytes(buffer.read())

    def plot_lineplot(self, x_column_name: str | None = None, y_column_name: str | None = None) -> Image:
        """

        Plot the time series target or the given column(s) as line plot.

        The function will take the time column as the default value for y_column_name and the target column as the
        default value for x_column_name.

        Parameters
        ----------
        x_column_name:
            The column name of the column to be plotted on the x-Axis, default is the time column.
        y_column_name:
            The column name of the column to be plotted on the y-Axis, default is the target column.

        Returns
        -------
        plot:
            The plot as an image.

        Raises
        ------
        NonNumericColumnError
            If the time series given columns contain non-numerical values.

        UnknownColumnNameError
            If one of the given names does not exist in the table

        Examples
        --------
                >>> from safeds.data.tabular.containers import TimeSeries
                >>> table = TimeSeries({"time":[1, 2], "target": [3, 4], "feature":[2,2]}, target_name= "target", time_name="time", feature_names=["feature"], )
                >>> image = table.plot_lineplot()

        """
        self._data.index.name = "index"
        if x_column_name is not None and not self.get_column(x_column_name).type.is_numeric():
            raise NonNumericColumnError("The time series plotted column contains non-numerical columns.")

        if y_column_name is None:
            y_column_name = self._target.name

        elif y_column_name not in self._data.columns:
            raise UnknownColumnNameError([y_column_name])

        if x_column_name is None:
            x_column_name = self.time.name

        if not self.get_column(y_column_name).type.is_numeric():
            raise NonNumericColumnError("The time series plotted column contains non-numerical columns.")

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
        self._data = self._data.reset_index()
        return Image.from_bytes(buffer.read())

    def plot_scatterplot(
        self,
        x_column_name: str | None = None,
        y_column_name: str | None = None,
    ) -> Image:
        """
        Plot the time series target or the given column(s) as scatter plot.

        The function will take the time column as the default value for x_column_name and the target column as the
        default value for y_column_name.

        Parameters
        ----------
        x_column_name:
            The column name of the column to be plotted on the x-Axis.
        y_column_name:
            The column name of the column to be plotted on the y-Axis.

        Returns
        -------
        plot:
            The plot as an image.

        Raises
        ------
        NonNumericColumnError
            If the time series given columns contain non-numerical values.

        UnknownColumnNameError
            If one of the given names does not exist in the table

        Examples
        --------
                >>> from safeds.data.tabular.containers import TimeSeries
                >>> table = TimeSeries({"time":[1, 2], "target": [3, 4], "feature":[2,2]}, target_name= "target", time_name="time", feature_names=["feature"], )
                >>> image = table.plot_scatterplot()

        """
        self._data.index.name = "index"
        if x_column_name is not None and not self.get_column(x_column_name).type.is_numeric():
            raise NonNumericColumnError("The time series plotted column contains non-numerical columns.")

        if y_column_name is None:
            y_column_name = self._target.name
        elif y_column_name not in self._data.columns:
            raise UnknownColumnNameError([y_column_name])
        if x_column_name is None:
            x_column_name = self.time.name

        if not self.get_column(y_column_name).type.is_numeric():
            raise NonNumericColumnError("The time series plotted column contains non-numerical columns.")

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
        self._data = self._data.reset_index()
        return Image.from_bytes(buffer.read())
