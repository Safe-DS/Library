from __future__ import annotations

from typing import Any

import pandas as pd
from safe_ds.exceptions import ColumnSizeError, IndexOutOfBoundsError

from ._column_type import ColumnType


class Column:
    def __init__(self, data: pd.Series, name: str) -> None:
        self._data: pd.Series = data
        self._name: str = name
        self._type: ColumnType = ColumnType.from_numpy_dtype(self._data.dtype)

    @property
    def name(self) -> str:
        """
        Get the name of the Column

        Returns
        -------
        name: str
            The name of the column
        """
        return self._name

    @property
    def type(self) -> ColumnType:
        """
        Get the type of the Column

        Returns
        -------
        type: ColumnType
            The type of the column
        """
        return self._type

    def __getitem__(self, index: int) -> Any:
        return self.get_value(index)

    def get_value(self, index: int) -> Any:
        """
        Returns column value at specified index, starting at 0.

        Parameters
        ----------
        index : int
            Index of requested element as integer.

        Returns
        -------
        value
            Value at index in column.

        Raises
        ------
        IndexOutOfBoundsError
            If the given index does not exist in the column.
        """
        if index < 0 or index >= self._data.size:
            raise IndexOutOfBoundsError(index)

        return self._data[index]

    def idness(self) -> float:
        """
        Calculates the idness of this column (number of unique values / number of rows).

        Returns
        -------
        idness: float
            The idness of the column
        """
        if self._data.size == 0:
            raise ColumnSizeError("> 0", "0")
        return self._data.nunique() / self._data.size

    @property
    def statistics(self) -> ColumnStatistics:
        return ColumnStatistics(self)

    def count_null_values(self) -> int:
        """
        Returns the number of null values in the column.

        Returns
        -------
        count : int
            Number of null values
        """
        return self._data.isna().sum()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Column):
            return NotImplemented
        if self is other:
            return True
        return self._data.equals(other._data) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self._data)


class ColumnStatistics:
    def __init__(self, column: Column):
        self._column = column

    def max(self) -> float:
        """
        Returns the maximum value of the column.

        Returns
        -------
        max:
            the maximum value

        Raises
        ------
        TypeError
            If the data contains non-numerical data.
        """
        if not self._column._type.is_numeric():
            raise TypeError("The column contains non numerical data.")
        return self._column._data.max()

    def min(self) -> float:
        """
        Returns the minimum value of the column.

        Returns
        -------
        min:
            the minimum value

        Raises
        ------
        TypeError
            If the data contains non-numerical data.
        """
        if not self._column._type.is_numeric():
            raise TypeError("The column contains non numerical data.")
        return self._column._data.min()

    def mean(self) -> float:
        """
        Returns the mean value of the column.

        Returns
        -------
        mean:
            the mean value

        Raises
        ------
        TypeError
            If the data contains non-numerical data.
        """
        if not self._column._type.is_numeric():
            raise TypeError("The column contains non numerical data.")
        return self._column._data.mean()

    def mode(self) -> Any:
        """
        Returns the mode of the column.

        Returns
        -------
        mode:
            the mode
        """
        return self._column._data.mode()[0]

    def median(self) -> float:
        """
        Returns the median value of the column.

        Returns
        -------
        median:
            the median value

        Raises
        ------
        TypeError
            If the data contains non-numerical data.
        """
        if not self._column._type.is_numeric():
            raise TypeError("The column contains non numerical data.")
        return self._column._data.median()
