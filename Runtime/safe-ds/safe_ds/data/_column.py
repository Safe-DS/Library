from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from safe_ds.exceptions import (
    ColumnLengthMismatchError,
    ColumnSizeError,
    IndexOutOfBoundsError,
    NonNumericColumnError,
)

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

        Raises
        ------
        ColumnSizeError
            If this column is empty
        """
        if self._data.size == 0:
            raise ColumnSizeError("> 0", "0")
        return self._data.nunique() / self._data.size

    @property
    def statistics(self) -> ColumnStatistics:
        return ColumnStatistics(self)

    def _count_missing_values(self) -> int:
        """
        Returns the number of null values in the column.

        Returns
        -------
        count : int
            Number of null values
        """
        return self._data.isna().sum()

    def all(self, predicate: Callable[[Any], bool]) -> bool:
        """
        Checks if all values have a given property

        Parameters
        ----------
        predicate: Callable[[Any], bool])
            Callable that is used to find matches

        Returns
        -------
        result: bool
            True if all match

        """
        for value in self._data:
            if not predicate(value):
                return False
        return True

    def any(self, predicate: Callable[[Any], bool]) -> bool:
        """
        Checks if any value has a given property

        Parameters
        ----------
        predicate: Callable[[Any], bool])
            Callable that is used to find matches

        Returns
        -------
        result: bool
            True if any match

        """
        for value in self._data:
            if predicate(value):
                return True
        return False

    def none(self, predicate: Callable[[Any], bool]) -> bool:
        """
        Checks if no values has a given property

        Parameters
        ----------
        predicate: Callable[[Any], bool])
            Callable that is used to find matches

        Returns
        -------
        result: bool
            True if none match

        """
        for value in self._data:
            if predicate(value):
                return False
        return True

    def missing_value_ratio(self) -> float:
        """
        Returns the ratio of null values to the total number of elements in the column

        Returns
        -------
        ratio: float
            the ratio of null values to the total number of elements in the column
        """
        if self._data.size == 0:
            raise ColumnSizeError("> 0", "0")
        return self._count_missing_values() / self._data.size

    def has_missing_values(self) -> bool:
        """
        Returns True if the column has missing values

        Returns
        -------
        : bool
            True if missing values exist, False else
        """
        return self.any(lambda value: value is None or np.isnan(value))

    def stability(self) -> float:
        """
        Calculates the stability of this column.
        The value is calculated as the ratio between the number of mode values and the number of non-null-values.

        Returns
        -------
        stability: float
            Stability of this column

        Raises
        ------
        ColumnSizeError
            If this column is empty
        """
        if self._data.size == 0:
            raise ColumnSizeError("> 0", "0")
        return self._data.value_counts()[self.statistics.mode()] / self._data.count()

    def correlation_with(self, other_column: Column) -> float:
        """
        Calculates Pearson correlation between this and another column, if both are numerical

        Returns
        -------
        correlation: float
            Correlation between the two columns

        Raises
        ------
        TypeError
            If one of the columns is not numerical
        """
        if not self._type.is_numeric() or not other_column._type.is_numeric():
            raise NonNumericColumnError(
                f"Columns must be numerical. {self.name} is {self._type}, "
                f"{other_column.name} is {other_column._type}."
            )
        if self._data.size != other_column._data.size:
            raise ColumnLengthMismatchError(
                f"{self.name} is of size {self._data.size}, "
                f"{other_column.name} is of size {other_column._data.size}."
            )
        return self._data.corr(other_column._data)

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
        self.column = column

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
        if not self.column._type.is_numeric():
            raise TypeError("The column contains non numerical data.")
        return self.column._data.max()

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
        if not self.column._type.is_numeric():
            raise TypeError("The column contains non numerical data.")
        return self.column._data.min()

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
        if not self.column._type.is_numeric():
            raise TypeError("The column contains non numerical data.")
        return self.column._data.mean()

    def mode(self) -> Any:
        """
        Returns the mode of the column.

        Returns
        -------
        mode:
            the mode
        """
        return self.column._data.mode()[0]

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
        if not self.column._type.is_numeric():
            raise TypeError("The column contains non numerical data.")
        return self.column._data.median()

    def sum(self) -> float:
        """
        Returns the sum of a numerical Column

        Returns
        -------
        sum:
            the sum of all values

        Raises
        ---
        NonNumericalColumnError
            If the data is non numerical

        """
        if not self.column.type.is_numeric():
            raise NonNumericColumnError(
                f"{self.column.name} is of type {self.column._type}."
            )
        return self.column._data.sum()

    def variance(self) -> float:

        """
        Returns the variance of a numerical Column

        Returns
        -------
        sum:
            the variance of all values

        Raises
        ---
        NonNumericalColumnError
            If the data is non numerical

        """
        if not self.column.type.is_numeric():
            raise NonNumericColumnError(
                f"{self.column.name} is of type {self.column._type}."
            )

        return self.column._data.var()

    def standard_deviation(self) -> float:

        """
        Returns the standard deviation of a numerical Column

        Returns
        -------
        sum:
            the standard deviation of all values

        Raises
        ---
        NonNumericalColumnError
            If the data is non numerical

        """
        if not self.column.type.is_numeric():
            raise NonNumericColumnError(
                f"{self.column.name} is of type {self.column._type}."
            )
        return self.column._data.std()
