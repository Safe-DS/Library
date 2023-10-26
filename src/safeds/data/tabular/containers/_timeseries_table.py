from __future__ import annotations

import copy
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

from safeds.data.tabular.containers import Column, Row, Table, TaggedTable
from safeds.exceptions import (
    ColumnIsTargetError,
    IllegalSchemaModificationError,
    UnknownColumnNameError,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Any


class TimeSeries(Table):
    """
    A TimeSeries is a TaggedTable, which has also a time column

        Parameters
    ----------
    data : Mapping[str, Sequence[Any]]
        The data.
    target_name : str
        Name of the target column.
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



    """

    # ------------------------------------------------------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _from_table(
        table: Table,
        target_name: str,
        date_name: str,
        window_size: int,
        forecast_horizon: int,
        feature_names: list[str] | None = None,
    ) -> TimeSeries:
        """Create a TimeSeries from a table

        Parameters
        ----------
        table : Table
            The table.
        target_name : str
            Name of the target column.
        date_name: str
            Name of the date column.
        window_size: int
            Size of the windows that will be created
        forecast_horizon: int
            Size of the forecast horizon
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
        >>> table = Table({"date": [0, 1, 2, 4]}, "f1": [5, 6, 7, 8])})
        >>> timeseries = TimeSeries._from_table(table, "f1", "date", 2, 1, ["f1"])


        """
        table = table._as_table()
        # check if target column exists
        if target_name not in table.column_names:
            raise UnknownColumnNameError([target_name])

        # If no feature names are specified, use all columns except the target column
        if feature_names is None:
            feature_names = table.column_names
            # still needs to be thought about, if a target can be a feature normally yes
            # feature_names.remove(target_name)
        if len(feature_names) == 0:
            raise ValueError("At least one feature column must be specified.")

        # Create Time Series Object
        result = object.__new__(TimeSeries)
        result._data = table._data
        result._schema = table._schema
        result._features = table.keep_only_columns(feature_names)
        result._target = table.get_column(target_name)
        result._date = table.get_column(date_name)
        result._window_size = window_size
        result._forecast_horizon = forecast_horizon
        return result

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        data: Mapping[str, Sequence[Any]],
        date_name: str,
        target_name: str,
        window_size: int,
        forecast_horizon: int,
        feature_names: list[str] | None = None,
    ):
        """
        Create a timeseries from a mapping of column names to their values
        A Timeseries is a tagged table

                Parameters
        ----------
        data : Mapping[str, Sequence[Any]]
            The data.
        date_name : str
            Name of the date column. 
        target_name : str
            Name of the target column.
        window_size : int
            Size of the created windows
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
        ------
        >>> from safeds.data.tabular.containers import TimeSeries
        >>> timeseries = TimeSeries({"date":[0, 1, 2, 3, 4], "f1":[1, 2, 3, 4, 5]}, "f1", "date", "f1", 2, 1, ["f1"] )
        """
        super().__init__(data)
        _data = Table(data)

        # If no feature names are specified, use all columns except the target column
        if feature_names is None:
            feature_names = _data.column_names
            if target_name in feature_names:
                feature_names.remove(target_name)

        # Validate inputs
        # if target_name in feature_names:
        #   raise ValueError(f"Column '{target_name}' cannot be both feature and target.")
        if len(feature_names) == 0:
            raise ValueError("At least one feature column must be specified.")

        self._features: Table = _data.keep_only_columns(feature_names)
        self._target: Column = _data.get_column(target_name)
        self._date: Column = _data.get_column(date_name)
        self._window_size: int = window_size
        self._forecast_horizon: int = forecast_horizon
        self._feature_names = feature_names
        self._target_name = target_name



        for col in self._create_all_windows_for_column():
            print(col)

        print(self._create_all_labels_for_target_column())
        
    
    def _create_all_windows_for_column(self):
    #this generator generates all windows for all feature columns
        def in_yield(col: Column):
            ser = col._data
            for i in range(len(ser) - self._window_size):
                yield ser.iloc[i : i + self._window_size]
        for col_name in self._feature_names:
            print("windowing feature col:" + col_name)
            col = self._features.get_column(col_name)
            yield list(in_yield(col)) 


    
    def _create_all_labels_for_target_column(self):
    #this generator generates all forecast horizons for the target column
        print("creating labels for target column:" + self.target.name)
        def _generate_label_windows( ):
            ser = self._target._data
            for i in range(len(ser) - self._forecast_horizon):
                yield ser.iloc[i + self._window_size : i + self._window_size + self._forecast_horizon]
        return list(_generate_label_windows())


    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

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
    def target(self) -> Column:
        """
        Get the target column of the tagged table.

        Returns
        -------
        Column
            The target column.
        """
        return self._target



    # ------------------------------------------------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------------------------------------------------
    def _copy(self) -> TimeSeries:
        """
        Return a copy of this time series

        Returns
        -------
        table: TimeSeries
        """  
        return copy.deepcopy(self)

    
    # ------------------------------------------------------------------------------------------------------------------
    # Specific methods from TimeSeries class:
    # ------------------------------------------------------------------------------------------------------------------

    def add_column_as_feature(self, column: Column) -> TaggedTable:
        """
        Return a new time series with the provoded column attached at the end, as a feature column.

        the original table is not modified
        
        Parameters
        ----------
        column : Column
            The  column to be added

        Returns
        -------
        result : TaggedTable
            The table with the attached feature column.
        
        Raises
        ------
        DuplicateColumnNameError
            If the new column already exists.
        ColumnSizeError
            If the size of the column does not match the number of rows.
        """
        return TimeSeries._from_table(
        super().add_column(column),
        target_name=self.target.name,
        date_name=  self._date.name,
        window_size=self._window_size,
        forecast_horizon=self.forecast_horizon,
        feature_names=[*self.features.column_names, column.name],
    )
    def add_columns_as_features(self, columns: list[Column] | Table) -> TaggedTable:
        """
        Return a new `TaggedTable` with the provided columns attached at the end, as feature columns.

        The original table is not modified.

        Parameters
        ----------
        columns : list[Column] | Table
            The columns to be added as features.

        Returns
        -------
        result : TaggedTable
            The table with the attached feature columns.

        Raises
        ------
        DuplicateColumnNameError
            If any of the new feature columns already exist.
        ColumnSizeError
            If the size of any feature column does not match the number of rows.
        """
        return TimeSeries._from_table(
            super().add_columns(columns),
            target_name= self.target.name,
            date_name = self._date.name,
            window_size= self._window_size,
            forecast_horizon=self._forecast_horizon,
            feature_names=self.features.column_names
            + [col.name for col in (columns.to_columns() if isinstance(columns, Table) else columns)],
        )
    
    def _as_table(self: TaggedTable) -> Table:
        """
        Return a new `Table` with the tagging removed.

        The original TaggedTable is not modified.

        Parameters
        ----------
        self: TimeSeries
            The TimeSeries.

        Returns
        -------
        table: Table
            The table as an untagged Table, i.e. without the information about which columns are features or target.

        """
        return Table.from_columns(super().to_columns())
    
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
        table : TaggedTable
            A new table with the added row at the end.

        Raises
        ------
        UnknownColumnNameError
            If the row has different column names than the table.
        """
        return TimeSeries._from_table(super().add_row(row),
                                       target_name=self.target.name,
                                       date_name= self._date.name,
                                       window_size=self._window_size,
                                       forecast_horizon=self._forecast_horizon,
                                       feature_names=self._feature_names)
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
            A new table which combines the original table and the given rows.

        Raises
        ------
        UnknownColumnNameError
            If at least one of the rows have different column names than the table.
        """
       
        return TimeSeries._from_table(super().add_row(rows),
                                       target_name=self.target.name,
                                       date_name= self._date.name,
                                       window_size=self._window_size,
                                       forecast_horizon=self._forecast_horizon,
                                       feature_names=self._feature_names)
    
    def filter_rows(self, query: Callable[[Row], bool]) -> TimeSeries:
            """
            Return a new `TimeSeries` containing only rows that match the given Callable (e.g. lambda function).

            The original table is not modified.

            Parameters
            ----------
            query : lambda function
                A Callable that is applied to all rows.

            Returns
            -------
            table : TaggedTable
                A table containing only the rows to match the query.
            """
            return TimeSeries._from_table(
                super().filter_rows(query),
                target_name=self.target.name,
                date_name=self._date.name,
                window_size= self._window_size,
                forecast_horizon=self._forecast_horizon,
                feature_names=self.features.column_names,
            )
    def _as_table(self: TimeSeries) -> Table:
        """
        Return a new `Table` with the tagging removed.

        The original TimeSeries is not modified.

        Parameters
        ----------
        self: TimeSeries
            The TimeSeries.

        Returns
        -------
        table: Table
            The table as an untagged Table, i.e. without the information about which columns are features or target.

        """
        return Table.from_columns(super().to_columns())
    
