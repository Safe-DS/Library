from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
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



class TimeSeries(TaggedTable):
    # target should always be an feature
    # date should never be an feature
    """
     A TimeSeries is a tagged table that additionally knows which column is the time column 
     and uses the target column as an feature.
     A Time Column should neve be an feature
    ----------
    data : Mapping[str, Sequence[Any]]
        The data.
    target_name : str
        Name of the target column.
    feature_names : list[str] | None
        Names of the feature columns. If None, all columns except the target column are used.
        Raises
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
        """Create a time series from a tagged table
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
        >>> timeseries = TimeSeries._from_table(tagged_table, "date")
        """
        table = tagged_table._as_table()

        #no need to validate the tagged table on tagged table specs
        if time_name in tagged_table.features.column_names:
            raise ValueError(f"Column '{time_name}' cannot be both time column and feature.")
        
        if time_name == tagged_table.target.name:
            raise ValueError(f"Column '{time_name}' cannot be both timecolumn and target.")
        
        
        # Create Time Series Object
        result = object.__new__(TimeSeries)

        result._data = table._data
        result._schema = table.schema
        result._time = table.get_column(time_name)
        result._features = table.keep_only_columns(tagged_table.features.column_names)
        result._target = table.get_column(tagged_table.target.name)
        return result

    @staticmethod
    def _from_table(
        table: Table,
        target_name: str,
        time_name: str,
        feature_names: list[str] | None = None,
    ) -> TimeSeries:
        """Create a TimeSeries from a table
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
        >>> table = Table({"date": ["01.01", "01.02", "01.03", "01.04"], "col1": ["a", "b", "c", "a"]})
        >>> timeseries = TimeSeries._from_table(table, "f1", "date", ["f1"])
        """

        tagged_table = TaggedTable._from_table(table=table, target_name= target_name, feature_names=feature_names)

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
                
        if time_name in feature_names:
            raise ValueError(f"Date column '{time_name}' can not be an feature column.")

        super().__init__(data, target_name, feature_names)
        _data = Table(data)

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
    
    #ToDo: add Specific methods for time series dTable class

    # ------------------------------------------------------------------------------------------------------------------
    # Overriden methods from TaggedTable class:
    # ------------------------------------------------------------------------------------------------------------------
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
            time_name = self.time,
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
            time_name= self.time.name,
        )