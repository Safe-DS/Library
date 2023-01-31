from __future__ import annotations

import typing
from numbers import Number
from typing import Any, Callable

import numpy as np
import pandas as pd
from IPython.core.display_functions import DisplayHandle, display
from safeds.data.tabular.typing import ColumnType
from safeds.exceptions import (
    ColumnLengthMismatchError,
    ColumnSizeError,
    IndexOutOfBoundsError,
    NonNumericColumnError,
)


class Column:
    def __init__(self, data: typing.Iterable, name: str) -> None:
        self._data: pd.Series = data if isinstance(data, pd.Series) else pd.Series(data)
        self._name: str = name
        self._type: ColumnType = ColumnType.from_numpy_dtype(self._data.dtype)

    @property
    def name(self) -> str:
        """
        Return the name of the column.

        Returns
        -------
        name : str
            The name of the column.
        """
        return self._name

    @property
    def type(self) -> ColumnType:
        """
        Return the type of the column.

        Returns
        -------
        type : ColumnType
            The type of the column.
        """
        return self._type

    def __getitem__(self, index: int) -> Any:
        return self.get_value(index)

    def get_value(self, index: int) -> Any:
        """
        Return column value at specified index, starting at 0.

        Parameters
        ----------
        index : int
            Index of requested element.

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

    @property
    def statistics(self) -> ColumnStatistics:
        return ColumnStatistics(self)

    def count(self) -> int:
        """
        Return the number of elements in the column.

        Returns
        -------
        count : int
            The number of elements.
        """
        return len(self._data)

    def _count_missing_values(self) -> int:
        """
        Return the number of null values in the column.

        Returns
        -------
        count : int
            The number of null values.
        """
        return self._data.isna().sum()

    def all(self, predicate: Callable[[Any], bool]) -> bool:
        """
        Check if all values have a given property.

        Parameters
        ----------
        predicate : Callable[[Any], bool])
            Callable that is used to find matches.

        Returns
        -------
        result : bool
            True if all match.

        """
        for value in self._data:
            if not predicate(value):
                return False
        return True

    def any(self, predicate: Callable[[Any], bool]) -> bool:
        """
        Check if any value has a given property.

        Parameters
        ----------
        predicate : Callable[[Any], bool])
            Callable that is used to find matches.

        Returns
        -------
        result : bool
            True if any match.

        """
        for value in self._data:
            if predicate(value):
                return True
        return False

    def none(self, predicate: Callable[[Any], bool]) -> bool:
        """
        Check if no values has a given property.

        Parameters
        ----------
        predicate : Callable[[Any], bool])
            Callable that is used to find matches.

        Returns
        -------
        result : bool
            True if none match.

        """
        for value in self._data:
            if predicate(value):
                return False
        return True

    def missing_value_ratio(self) -> float:
        """
        Return the ratio of null values to the total number of elements in the column

        Returns
        -------
        ratio : float
            The ratio of null values to the total number of elements in the column.
        """
        if self._data.size == 0:
            raise ColumnSizeError("> 0", "0")
        return self._count_missing_values() / self._data.size

    def has_missing_values(self) -> bool:
        """
        Return whether the column has missing values.

        Returns
        -------
        missing_values_exist : bool
            True if missing values exist.
        """
        return self.any(
            lambda value: value is None
            or (isinstance(value, Number) and np.isnan(value))
        )

    def correlation_with(self, other_column: Column) -> float:
        """
        Calculate Pearson correlation between this and another column. Both columns have to be numerical.

        Returns
        -------
        correlation : float
            Correlation between the two columns.

        Raises
        ------
        TypeError
            If one of the columns is not numerical.
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

    def get_unique_values(self) -> list[typing.Any]:
        """
        Return a list of all unique values in the column.

        Returns
        -------
        unique_values : list[any]
            List of unique values in the column.
        """
        return list(self._data.unique())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Column):
            return NotImplemented
        if self is other:
            return True
        return self._data.equals(other._data) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self._data)

    def __str__(self) -> str:
        tmp = self._data.to_frame()
        tmp.columns = [self.name]
        return tmp.__str__()

    def __repr__(self) -> str:
        tmp = self._data.to_frame()
        tmp.columns = [self.name]
        return tmp.__repr__()

    def _ipython_display_(self) -> DisplayHandle:
        """
        Return a display object for the column to be used in Jupyter Notebooks.

        Returns
        -------
        output : DisplayHandle
            Output object.
        """
        tmp = self._data.to_frame()
        tmp.columns = [self.name]

        with pd.option_context(
            "display.max_rows", tmp.shape[0], "display.max_columns", tmp.shape[1]
        ):
            return display(tmp)


class ColumnStatistics:
    def __init__(self, column: Column):
        self._column = column

    def max(self) -> float:
        """
        Return the maximum value of the column. The column has to be numerical.

        Returns
        -------
        max : float
            The maximum value.

        Raises
        ------
        NonNumericColumnError
            If the data contains non-numerical data.
        """
        if not self._column._type.is_numeric():
            raise NonNumericColumnError(
                f"{self._column.name} is of type {self._column._type}."
            )
        return self._column._data.max()

    def min(self) -> float:
        """
        Return the minimum value of the column. The column has to be numerical.

        Returns
        -------
        min : float
            The minimum value.

        Raises
        ------
        NonNumericColumnError
            If the data contains non-numerical data.
        """
        if not self._column._type.is_numeric():
            raise NonNumericColumnError(
                f"{self._column.name} is of type {self._column._type}."
            )
        return self._column._data.min()

    def mean(self) -> float:
        """
        Return the mean value of the column. The column has to be numerical.

        Returns
        -------
        mean : float
            The mean value.

        Raises
        ------
        NonNumericColumnError
            If the data contains non-numerical data.
        """
        if not self._column._type.is_numeric():
            raise NonNumericColumnError(
                f"{self._column.name} is of type {self._column._type}."
            )
        return self._column._data.mean()

    def mode(self) -> Any:
        """
        Return the mode of the column.

        Returns
        -------
        List :
            Returns a list with the most common values.
        """
        return self._column._data.mode().tolist()

    def median(self) -> float:
        """
        Return the median value of the column. The column has to be numerical.

        Returns
        -------
        median : float
            The median value.

        Raises
        ------
        NonNumericColumnError
            If the data contains non-numerical data.
        """
        if not self._column._type.is_numeric():
            raise NonNumericColumnError(
                f"{self._column.name} is of type {self._column._type}."
            )
        return self._column._data.median()

    def sum(self) -> float:
        """
        Return the sum of the column. The column has to be numerical.

        Returns
        -------
        sum : float
            The sum of all values.

        Raises
        ---
        NonNumericColumnError
            If the data contains non-numerical data.

        """
        if not self._column.type.is_numeric():
            raise NonNumericColumnError(
                f"{self._column.name} is of type {self._column._type}."
            )
        return self._column._data.sum()

    def variance(self) -> float:

        """
        Return the variance of the column. The column has to be numerical.

        Returns
        -------
        sum : float
            The variance of all values.

        Raises
        ---
        NonNumericColumnError
            If the data contains non-numerical data.

        """
        if not self._column.type.is_numeric():
            raise NonNumericColumnError(
                f"{self._column.name} is of type {self._column._type}."
            )

        return self._column._data.var()

    def standard_deviation(self) -> float:

        """
        Return the standard deviation of the column. The column has to be numerical.

        Returns
        -------
        sum : float
            The standard deviation of all values.

        Raises
        ---
        NonNumericColumnError
            If the data contains non-numerical data.

        """
        if not self._column.type.is_numeric():
            raise NonNumericColumnError(
                f"{self._column.name} is of type {self._column._type}."
            )
        return self._column._data.std()

    def stability(self) -> float:
        """
        Calculate the stability of this column, which we define as

        $$
        \\frac{\\text{number of occurrences of most common non-null value}}{\\text{number of non-null values}}
        $$

        Returns
        -------
        stability : float
            The stability of the column.

        Raises
        ------
        ColumnSizeError
            If the column is empty.
        """
        if self._column._data.size == 0:
            raise ColumnSizeError("> 0", "0")
        return (
            self._column._data.value_counts()[self._column.statistics.mode()[0]]
            / self._column._data.count()
        )

    def idness(self) -> float:
        """
        Calculate the idness of this column, which we define as

        $$
        \\frac{\\text{number of different values}}{\\text{number of rows}}
        $$

        Returns
        -------
        idness : float
            The idness of the column.

        Raises
        ------
        ColumnSizeError
            If this column is empty.
        """
        if self._column._data.size == 0:
            raise ColumnSizeError("> 0", "0")
        return self._column._data.nunique() / self._column._data.size
