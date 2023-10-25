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
        date_name: str,
        window_size: int,
        feature_names: list[str] | None = None,
    ) -> TimeSeries:
        """Create a TimeSeries from a table
        
        Parameters
        ----------
        table : Table
            The table.
        target_name : str
            Name of the target column.
        feature_names : list[str] | None
            Names of the feature columns. If None, all columns except the target column are used.

        
        """
        table = table._as_table()
        #check if target column exists
        if target_name not in table.column_names:
            raise UnknownColumnNameError([target_name])
        
        # If no feature names are specified, use all columns except the target column
        if feature_names is None:
            feature_names = table.column_names
            # still needs to be thought about, if a target can be a feature normally yes
            #feature_names.remove(target_name)
        if len(feature_names) == 0:
            raise ValueError("At least one feature column must be specified.")
        
        # Create Time Series Object
        result = object.__new__(TimeSeries)
        result._data = table._data
        result._schema = table._schema
        result._features = table.keep_only_columns(feature_names)
        result._target = table.get_column(target_name)
        result._date = table.get_column(date_name)

        #window the given data
        result._features = result._make_windowed_table_for_features()
        result._target = result._make_windows(result._target)
        return result

    

        


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
        # but the data gets reshaped alot without the user knowing


        #window the given data
        self._features = self._make_windowed_table_for_features()
        self._target = self._make_target_windows()


    def _make_windowed_table_for_features(self):
        #init list, to create a Table out of a list of columns
        col_list = []
        for col_name in self._features.column_names:
            #get the column
            temp_col = self._features.get_column(col_name)
            #get data of the column and load it into a numpy array, also create windowed numpy array
            x_s = self._make_windows(temp_col._data.to_numpy())
            #load the new data in
            col_list.append(Column._from_pandas_series(pd.Series(x_s, name = col_name)))
        return Table.from_columns(col_list)

    
    def _make_target_windows(self):
        seq = self._target._data.to_numpy()
        y_s = []
        l = len(seq)
        for i in range(l - self._window_size):
            # its sliced so its an np.ndarray
            # we mby also need longer arrays then 1 for forecast horizon
            label = seq[i+self._window_size:i+self._window_size+1]
            y_s.append(label)
        print(y_s)
        self._target = Column._from_pandas_series(pd.Series(y_s, name = self._target.name))

    #toDo: Change behavior of this function so it behacios similiar to the other window functions
    #      it is just a subrouting of _make_windowed_table_for_features
    def _make_windows(self, seq):
        x_s = []
        l = len(seq)
        for i in range(l -self._window_size):
            window = seq[i:i+self._window_size]
            x_s.append(window)
        print(x_s)
        return x_s


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
