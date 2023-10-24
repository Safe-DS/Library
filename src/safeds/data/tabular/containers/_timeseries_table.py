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
    def _from_tagged_table(
        tagged_table: TaggedTable,
        time_column: str,
        window_size: int,
    )->TimeSeries:
        pass

    @staticmethod
    def _from_table(
        table: Table,
        target_name: str,
        feature_names: list[str] | None = None,
    ) -> TimeSeries:
        "Create a TimeSeries from a table"
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        data: Mapping[str, Sequence[Any]],
        target_name: str,
        date_name: str,
        window_size: int,
        feature_names: list[str] | None = None,
        ):
        """
        Create a timeseries from a mapping of column names to their values
        A Timeseries is a tagged table

                Parameters
        ----------
        data : Mapping[str, Sequence[Any]]
            The data.
        target_name : str
            Name of the target column.
        date_name : str
            Name of the date column.
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

        """
        print("hi")
        super().__init__(data)
        _data = Table(data)

        # If no feature names are specified, use all columns except the target column
        if feature_names is None:
            feature_names = _data.column_names
            if target_name in feature_names:
                feature_names.remove(target_name)

        # Validate inputs
        #if target_name in feature_names:
        #   raise ValueError(f"Column '{target_name}' cannot be both feature and target.")
        if len(feature_names) == 0:
            raise ValueError("At least one feature column must be specified.")

        self._features: Table = _data.keep_only_columns(feature_names)
        self._target: Column = _data.get_column(target_name)
        self._date: Column = _data.get_column(date_name)
        self._window_size: int = window_size

        # We have a Target Column and a Data Column
        # So we need a function which takes all features and also the target
        # but the data gets reshaped alot wo the user knowing
        column_list = []
        for col_name in self._features.column_names:
            ds_col = self._features.get_column(col_name)
            feature_column = ds_col._data.to_numpy()
            x_s = self._makeWindows(feature_column, window_size)
            #append column to column list
            col = Column._from_pandas_series(pd.Series(x_s, name = col_name),ds_col.type)
            column_list.append(col)

        self._features = Table.from_columns(column_list)
        self._target = self._makeWindows(self._target, window_size)



    def _makeWindows(self, seq, window_size: int):
        x_s = []
        y_s = []
        l = len(seq)
        for i in range(l - window_size-1):
            window = seq[i:i+window_size]
            label = seq[i + window_size + 1]
            x_s.append(window)
            y_s.append(label)
        return x_s, 



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
