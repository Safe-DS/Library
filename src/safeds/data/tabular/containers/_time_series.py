from __future__ import annotations

from typing import TYPE_CHECKING

from safeds.data.tabular.containers import Column, Row, Table, TaggedTable
from safeds.exceptions import (
    ColumnIsTargetError,
    ColumnIsTimeError,
    IllegalSchemaModificationError,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Any


class TimeSeries(TaggedTable):
    """A TimeSeries is a tagged table that additionally knows which column is the time column and uses the target column as an feature.

     A Time Column should never be an feature.
    ----------
    data : Mapping[str, Sequence[Any]]
        The data.
    target_name : str
        Name of the target column.
    feature_names : list[str] | None
        Names of the feature columns. If None, all columns except the target column are used.

    Raises
    ------
    time_name : str
        Name of the time column.

    ------
    ColumnLengthMismatchError
        If columns have different lengths.
    ValueError
        If the target column is also a feature column.
    ValueError
        If no feature columns are specified.
    """

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
        table : TaggedTable
            The tagged table.
        time_name: str
            Name of the time column.
        Retruns
        -------
        time_series : TimeSeries
            the created time series

        Raises
        ------
        UnknownColumnError
            If target_name matches none of the column names.
        Value Error
            If no feature columns are specified

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table, TimeSeries
        >>> tagged_table = TaggedTable({"date": ["01.01", "01.02", "01.03", "01.04"], "col1": ["a", "b", "c", "a"]}, "col1" )
        >>> timeseries = TimeSeries._from_tagged_table(tagged_table, time_name = "date")
        """
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
    # idk if this method should exist, because we cant use it like the Method in TaggedTable, but the Method is static and this seem to doesnt get checked by the megalinter.
    def _from_table_to_time_series(
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
            Names of the feature columns. If None, all columns except the target column are used.
        Retruns
        -------
        time_series : TimeSeries
            the created time series

        Raises
        ------
        UnknownColumnError
            If target_name matches none of the column names.
        Value Error
            If no feature columns are specified

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table, TimeSeries
        >>> table = Table({"date": ["01.01", "01.02", "01.03", "01.04"], "f1": ["a", "b", "c", "a"], "t": [1,2,3,4]})
        >>> timeseries = TimeSeries._from_table_to_time_series(table, "t", "date", ["f1"])
        """
        if feature_names is None:
            feature_names = table.column_names
            if time_name in feature_names:
                feature_names.remove(time_name)
            if target_name in feature_names:
                feature_names.remove(target_name)
        tagged_table = TaggedTable._from_table(table=table, target_name=target_name, feature_names=feature_names)
        # check if time column got added as feature column
        return TimeSeries._from_tagged_table(tagged_table=tagged_table, time_name=time_name)

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
        Create a tagged table from a mapping of column names to their values.

        Parameters
        ----------
        data : Mapping[str, Sequence[Any]]
            The data.
        target_name : str
            Name of the target column.
        time_name : str
            Name of the time column
        feature_names : list[str] | None
            Names of the feature columns. If None, all columns except the target column are used.

        Raises
        ------
        ColumnLengthMismatchError
            If columns have different lengths.
        ValueError
            If the target column is also a feature column.
        ValueError
            If no feature columns are specified.

        Examples
        --------
        >>> from safeds.data.tabular.containers import TaggedTable
        >>> table = TaggedTable({"a": [1, 2, 3], "b": [4, 5, 6]}, "b", ["a"])
        """
        _data = Table(data)
        # time sollte nicht in feature names sein auch wenn feature_names none ist
        if feature_names is None:
            feature_names = _data.column_names
            if time_name in feature_names:
                feature_names.remove(time_name)
            if target_name in feature_names:
                feature_names.remove(target_name)

        super().__init__(data, target_name, feature_names)

        # Validate inputs
        if time_name not in (_data.column_names):
            raise ValueError(f"Column '{time_name}' must exist in the table.")

        self._time: Column = _data.get_column(time_name)

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

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
    # Specific methods from time series class:
    # ------------------------------------------------------------------------------------------------------------------

    # def _create_moving_average(self, window_size: int) -> Column:
    # calculate the windows and the moving average of a timeserties, this function for now only has intern use
    # some unexpected behavior while casting to series agains, because it addds NAN values
    # return Column._from_pandas_series(pd.Series(self.target._data.rolling(window_size).mean()))
    # pass
    # def plot_moving_average(self, window_size: int) -> Image:
    # this method should plot the moving average of a time series
    # pass

    # def show_time_series(self) -> Image:
    # this method should plot the time series
    # pass

    # def decompose_time_series(self) -> Image:
    # this Method should show a plot of a decomposed time series
    # pass

    # ------------------------------------------------------------------------------------------------------------------
    # Overriden methods from TaggedTable class:
    # ------------------------------------------------------------------------------------------------------------------
    def _as_table(self: TimeSeries) -> Table:
        """
        Return a new `Table` with the tagging removed.

        The original TaggedTable is not modified.

        Parameters
        ----------
        self: TaggedTable
            The TaggedTable.

        Returns
        -------
        table: Table
            The table as an untagged Table, i.e. without the information about which columns are features or target.

        """
        return Table.from_columns(super().to_columns())

    def add_column(self, column: Column) -> TimeSeries:
        """
        Return a new `TimeSeries` with the provided column attached at the end, as neither target nor feature column.

        The original table is not modified.

        Parameters
        ----------
        column : Column
            The column to be added.

        Returns
        -------
        result : TimeSeries
            The table with the column attached as neither target nor feature column.

        Raises
        ------
        DuplicateColumnNameError
            If the new column already exists.
        ColumnSizeError
            If the size of the column does not match the number of rows.
        """
        return TimeSeries._from_tagged_table(
            super().add_column(column),
            time_name=self.time.name,
        )

    def add_column_as_feature(self, column: Column) -> TimeSeries:
        """
        Return a new time series with the provided column attached at the end, as a feature column.

        the original time series is not modified.

        Parameters
        ----------
        column : Column
            The column to be added.

        Returns
        -------
        result : TimeSeries
            The table with the attached feature column.

        Raises
        ------
        DuplicateColumnNameError
            If the new column already exists.
        ColumnSizeError
            If the size of the column does not match the number of rows.
        """
        return TimeSeries._from_tagged_table(
            super().add_column_as_feature(column),
            time_name=self.time.name,
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
        return TimeSeries._from_tagged_table(
            super().add_columns_as_features(columns),
            time_name=self.time.name,
        )

    def add_columns(self, columns: list[Column] | Table) -> TimeSeries:
        """
        Return a new `TimeSeries` with multiple added columns, as neither target nor feature columns.

        The original table is not modified.

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
            If at least one column name from the provided column list already exists in the table.
        ColumnSizeError
            If at least one of the column sizes from the provided column list does not match the table.
        """
        return TimeSeries._from_tagged_table(
            super().add_columns(columns),
            time_name=self.time.name,
        )

    def add_row(self, row: Row) -> TimeSeries:
        """
        Return a new `TimeSeries` with an added Row attached.

        The original table is not modified.

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
            If the row has different column names than the table.
        """
        return TimeSeries._from_tagged_table(super().add_row(row), time_name=self.time.name)

    def add_rows(self, rows: list[Row] | Table) -> TimeSeries:
        """
        Return a new `TimeSeries` with multiple added Rows attached.

        The original table is not modified.

        Parameters
        ----------
        rows : list[Row] or Table
            The rows to be added.

        Returns
        -------
        result : TimeSeries
            A new time series which combines the original table and the given rows.

        Raises
        ------
        UnknownColumnNameError
            If at least one of the rows have different column names than the table.
        """
        return TimeSeries._from_tagged_table(super().add_rows(rows), time_name=self.time.name)

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
        return TimeSeries._from_tagged_table(
            super().filter_rows(query),
            time_name=self.time.name,
        )

    def keep_only_columns(self, column_names: list[str]) -> TimeSeries:
        """
        Return a new `TimeSeries` with only the given column(s).

        The original table is not modified.

        Parameters
        ----------
        column_names : list[str]
            A list containing only the columns to be kept.

        Returns
        -------
        table : TimeSeries
            A time series containing only the given column(s).

        Raises
        ------
        UnknownColumnNameError
            If any of the given columns does not exist.
        IllegalSchemaModificationError
            If none of the given columns is the target column or any of the feature columns.
        """
        if self.target.name not in column_names:
            raise IllegalSchemaModificationError("Must keep the target column.")
        if len(set(self.features.column_names).intersection(set(column_names))) == 0:
            raise IllegalSchemaModificationError("Must keep at least one feature column.")
        if self.time.name not in column_names:
            raise IllegalSchemaModificationError("Must keep the time column.")
        return TimeSeries._from_tagged_table(
            TaggedTable._from_table(
                super().keep_only_columns(column_names),
                target_name=self.target.name,
                feature_names=sorted(
                    set(self.features.column_names).intersection(set(column_names)),
                    key={val: ix for ix, val in enumerate(self.features.column_names)}.__getitem__,
                ),
            ),
            time_name=self.time.name,
        )

    def remove_columns(self, column_names: list[str]) -> TimeSeries:
        """
        Return a new `TimeSeries` with the given column(s) removed from the table.

        The original table is not modified.

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
        IllegalSchemaModificationError
            If the given columns contain all the feature columns.
        """
        if self.target.name in column_names:
            raise ColumnIsTargetError(self.target.name)
        if len(set(self.features.column_names) - set(column_names)) == 0:
            raise IllegalSchemaModificationError("You cannot remove every feature column.")
        if self.time.name in column_names:
            raise ColumnIsTimeError(self.time.name)
        return TimeSeries._from_tagged_table(
            TaggedTable._from_table(
                super().remove_columns(column_names),
                target_name=self.target.name,
                feature_names=sorted(
                    set(self.features.column_names) - set(column_names),
                    key={val: ix for ix, val in enumerate(self.features.column_names)}.__getitem__,
                ),
            ),
            time_name=self.time.name,
        )

    def remove_columns_with_missing_values(self) -> TimeSeries:
        """
        Return a new `TimeSeries` with every column that misses values removed.

        The original table is not modified.

        Returns
        -------
        table : TimeSeries
            A time series without the columns that contain missing values.

        Raises
        ------
        ColumnIsTargetError
            If any of the columns to be removed is the target column.
        IllegalSchemaModificationError
            If the columns to remove contain all the feature columns.
        """
        table = super().remove_columns_with_missing_values()
        if self.target.name not in table.column_names:
            raise ColumnIsTargetError(self.target.name)
        if len(set(self.features.column_names).intersection(set(table.column_names))) == 0:
            raise IllegalSchemaModificationError("You cannot remove every feature column.")
        if self.time.name not in table.column_names:
            print("hi")
            raise ColumnIsTimeError(self.time.name)
        return TimeSeries._from_tagged_table(
            TaggedTable._from_table(
                table,
                self.target.name,
                feature_names=sorted(
                    set(self.features.column_names).intersection(set(table.column_names)),
                    key={val: ix for ix, val in enumerate(self.features.column_names)}.__getitem__,
                ),
            ),
            time_name=self.time.name,
        )

    def remove_columns_with_non_numerical_values(self) -> TimeSeries:
        """
        Return a new `TimeSeries` with every column that contains non-numerical values removed.

        The original table is not modified.

        Returns
        -------
        table : TimeSeries
            A time series without the columns that contain non-numerical values.

        Raises
        ------
        ColumnIsTargetError
            If any of the columns to be removed is the target column.
        IllegalSchemaModificationError
            If the columns to remove contain all the feature columns.
        """
        table = super().remove_columns_with_non_numerical_values()
        if self.target.name not in table.column_names:
            raise ColumnIsTargetError(self.target.name)
        if len(set(self.features.column_names).intersection(set(table.column_names))) == 0:
            raise IllegalSchemaModificationError("You cannot remove every feature column.")
        if self.time.name not in table.column_names:
            raise ColumnIsTimeError(self.time.name)
        return TimeSeries._from_tagged_table(
            TaggedTable._from_table(
                table,
                self.target.name,
                feature_names=sorted(
                    set(self.features.column_names).intersection(set(table.column_names)),
                    key={val: ix for ix, val in enumerate(self.features.column_names)}.__getitem__,
                ),
            ),
            time_name=self.time.name,
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
        return TimeSeries._from_tagged_table(
            TaggedTable._from_table(
                super().remove_duplicate_rows(),
                target_name=self.target.name,
                feature_names=self.features.column_names,
            ),
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
        return TimeSeries._from_tagged_table(
            TaggedTable._from_table(
                super().remove_rows_with_missing_values(),
                target_name=self.target.name,
                feature_names=self.features.column_names,
            ),
            time_name=self.time.name,
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
        return TimeSeries._from_tagged_table(
            TaggedTable._from_table(
                super().remove_rows_with_outliers(),
                target_name=self.target.name,
                feature_names=self.features.column_names,
            ),
            time_name=self.time.name,
        )

    def rename_column(self, old_name: str, new_name: str) -> TimeSeries:
        """
        Return a new `TimeSeries` with a single column renamed.

        The original time series is not modified.

        Parameters
        ----------
        old_name : str
            The old name of the target column.
        new_name : str
            The new name of the target column.

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
        return TimeSeries._from_tagged_table(
            TaggedTable._from_table(
                super().rename_column(old_name, new_name),
                target_name=new_name if self.target.name == old_name else self.target.name,
                feature_names=(
                    self.features.column_names
                    if old_name not in self.features.column_names
                    else [
                        column_name if column_name != old_name else new_name
                        for column_name in self.features.column_names
                    ]
                ),
            ),
            time_name=new_name if self.time.name == old_name else self.time.name,
        )

    def replace_column(self, old_column_name: str, new_columns: list[Column]) -> TimeSeries:
        """
        Return a new `TimeSeries` with the specified old column replaced by a list of new columns.

        If the column to be replaced is the target column, it must be replaced by exactly one column. That column
        becomes the new target column. If the column to be replaced is a feature column, the new columns that replace it
        all become feature columns.

        The order of columns is kept. The original table is not modified.

        Parameters
        ----------
        old_column_name : str
            The name of the column to be replaced.
        new_columns : list[Column]
            The new columns replacing the old column.

        Returns
        -------
        result : TimeSeries
            A time series with the old column replaced by the new column.

        Raises
        ------
        UnknownColumnNameError
            If the old column does not exist.
        DuplicateColumnNameError
            If the new column already exists and the existing column is not affected by the replacement.
        ColumnSizeError
            If the size of the column does not match the amount of rows.
        IllegalSchemaModificationError
            If the target column would be removed or replaced by more than one column.
        """
        if old_column_name == self.time.name:
            if len(new_columns) != 1:
                raise IllegalSchemaModificationError(
                    f'Time column "{self.time.name}" can only be replaced by exactly one new column.',
                )
            else:
                return TimeSeries._from_tagged_table(
                    TaggedTable._from_table(
                        super().replace_column(old_column_name, new_columns),
                        target_name=self.target.name,
                        feature_names=self.features.column_names,
                    ),
                    time_name=new_columns[0].name,
                )
        if old_column_name == self.target.name:
            if len(new_columns) != 1:
                raise IllegalSchemaModificationError(
                    f'Target column "{self.target.name}" can only be replaced by exactly one new column.',
                )
            else:
                return TimeSeries._from_tagged_table(
                    TaggedTable._from_table(
                        super().replace_column(old_column_name, new_columns),
                        target_name=new_columns[0].name,
                        feature_names=self.features.column_names,
                    ),
                    time_name=self.time.name,
                )
        else:
            return TimeSeries._from_tagged_table(
                TaggedTable._from_table(
                    super().replace_column(old_column_name, new_columns),
                    target_name=self.target.name,
                    feature_names=(
                        self.features.column_names
                        if old_column_name not in self.features.column_names
                        else self.features.column_names[: self.features.column_names.index(old_column_name)]
                        + [col.name for col in new_columns]
                        + self.features.column_names[self.features.column_names.index(old_column_name) + 1 :]
                    ),
                ),
                time_name=self.time.name,
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
        return TimeSeries._from_tagged_table(
            TaggedTable._from_table(
                super().slice_rows(start, end, step),
                target_name=self.target.name,
                feature_names=self.features.column_names,
            ),
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

        The original table is not modified.

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
        return TimeSeries._from_tagged_table(
            TaggedTable._from_table(
                sorted_table,
                target_name=self.target.name,
                feature_names=sorted(
                    set(sorted_table.column_names).intersection(self.features.column_names),
                    key={val: ix for ix, val in enumerate(sorted_table.column_names)}.__getitem__,
                ),
            ),
            time_name=self.time.name,
        )

    def transform_column(self, name: str, transformer: Callable[[Row], Any]) -> TimeSeries:
        """
        Return a new `TimeSeries` with the provided column transformed by calling the provided transformer.

        The original time series is not modified.

        Returns
        -------
        result : TimeSeries
            The time series with the transformed column.

        Raises
        ------
        UnknownColumnNameError
            If the column does not exist.
        """
        return TimeSeries._from_tagged_table(
            TaggedTable._from_table(
                super().transform_column(name, transformer),
                target_name=self.target.name,
                feature_names=self.features.column_names,
            ),
            time_name=self.time.name,
        )
