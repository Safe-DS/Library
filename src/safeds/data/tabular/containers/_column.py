from __future__ import annotations

import copy
import io
from collections.abc import Sequence
from numbers import Number
from typing import TYPE_CHECKING, Any, TypeVar, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from safeds.data.image.containers import Image
from safeds.data.image.typing import ImageFormat
from safeds.data.tabular.typing import ColumnType
from safeds.exceptions import (
    ColumnLengthMismatchError,
    ColumnSizeError,
    IndexOutOfBoundsError,
    NonNumericColumnError,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

T = TypeVar("T")
R = TypeVar("R")


class Column(Sequence[T]):
    """
    A column is a named collection of values.

    Parameters
    ----------
    name : str
        The name of the column.
    data : Sequence[T]
        The data.

    Examples
    --------
    >>> from safeds.data.tabular.containers import Column
    >>> column = Column("test", [1, 2, 3])
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _from_pandas_series(data: pd.Series, type_: ColumnType | None = None) -> Column:
        """
        Create a column from a `pandas.Series`.

        Parameters
        ----------
        data : pd.Series
            The data.
        type_ : ColumnType | None
            The type. If None, the type is inferred from the data.

        Returns
        -------
        column : Column
            The created column.

        Examples
        --------
        >>> import pandas as pd
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column._from_pandas_series(pd.Series([1, 2, 3], name="test"))
        """
        result = object.__new__(Column)
        result._name = data.name
        result._data = data
        # noinspection PyProtectedMember
        result._type = type_ if type_ is not None else ColumnType._data_type(data)

        return result

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, name: str, data: Sequence[T] | None = None) -> None:
        """
        Create a column.

        Parameters
        ----------
        name : str
            The name of the column.
        data : Sequence[T] | None
            The data. If None, an empty column is created.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3])
        """
        if data is None:
            data = []

        self._name: str = name
        self._data: pd.Series = data.rename(name) if isinstance(data, pd.Series) else pd.Series(data, name=name)
        # noinspection PyProtectedMember
        self._type: ColumnType = ColumnType._data_type(self._data)

    def __contains__(self, item: Any) -> bool:
        return item in self._data

    def __eq__(self, other: object) -> bool:
        """
        Check whether this column is equal to another object.

        Parameters
        ----------
        other : object
            The other object.

        Returns
        -------
        equal : bool
            True if the other object is an identical column. False if the other object is a different column.
            NotImplemented if the other object is not a column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("test", [1, 2, 3])
        >>> column2 = Column("test", [1, 2, 3])
        >>> column1 == column2
        True

        >>> column3 = Column("test", [3, 4, 5])
        >>> column1 == column3
        False
        """
        if not isinstance(other, Column):
            return NotImplemented
        if self is other:
            return True
        return self.name == other.name and self._data.equals(other._data)

    @overload
    def __getitem__(self, index: int) -> T:
        ...

    @overload
    def __getitem__(self, index: slice) -> Column[T]:
        ...

    def __getitem__(self, index: int | slice) -> T | Column[T]:
        """
        Return the value of the specified row or rows.

        Parameters
        ----------
        index : int | slice
            The index of the row, or a slice specifying the start and end index.

        Returns
        -------
        value : Any
            The single row's value, or rows' values.

        Raises
        ------
        IndexOutOfBoundsError
            If the given index or indices do not exist in the column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3])
        >>> column[0]
        1
        """
        if isinstance(index, int):
            if index < 0 or index >= self._data.size:
                raise IndexOutOfBoundsError(index)
            return self._data[index]

        if isinstance(index, slice):
            if index.start < 0 or index.start > self._data.size:
                raise IndexOutOfBoundsError(index)
            if index.stop < 0 or index.stop > self._data.size:
                raise IndexOutOfBoundsError(index)
            data = self._data[index].reset_index(drop=True).rename(self.name)
            return Column._from_pandas_series(data, self._type)

    def __iter__(self) -> Iterator[T]:
        r"""
        Create an iterator for the data of this column. This way e.g. for-each loops can be used on it.

        Returns
        -------
        iterator : Iterator[Any]
            The iterator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", ["A", "B", "C"])
        >>> string = ""
        >>> for val in column:
        ...     string += val + ", "
        >>> string
        'A, B, C, '
        """
        return iter(self._data)

    def __len__(self) -> int:
        """
        Return the size of the column.

        Returns
        -------
        n_rows : int
            The size of the column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3])
        >>> len(column)
        3
        """
        return len(self._data)

    def __repr__(self) -> str:
        """
        Return an unambiguous string representation of this column.

        Returns
        -------
        representation : str
            The string representation.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3])
        >>> repr(column)
        "Column('test', [1, 2, 3])"
        """
        return f"Column({self._name!r}, {list(self._data)!r})"

    def __str__(self) -> str:
        """
        Return a user-friendly string representation of this column.

        Returns
        -------
        representation : str
            The string representation.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3])
        >>> str(column)
        "'test': [1, 2, 3]"
        """
        return f"{self._name!r}: {list(self._data)!r}"

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def name(self) -> str:
        """
        Return the name of the column.

        Returns
        -------
        name : str
            The name of the column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3])
        >>> column.name
        'test'
        """
        return self._name

    @property
    def number_of_rows(self) -> int:
        """
        Return the number of elements in the column.

        Returns
        -------
        number_of_rows : int
            The number of elements.
        """
        return len(self._data)

    @property
    def type(self) -> ColumnType:
        """
        Return the type of the column.

        Returns
        -------
        type : ColumnType
            The type of the column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3])
        >>> column.type
        Integer

        >>> column = Column("test", ['a', 'b', 'c'])
        >>> column.type
        String
        """
        return self._type

    # ------------------------------------------------------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------------------------------------------------------

    def get_unique_values(self) -> list[T]:
        """
        Return a list of all unique values in the column.

        Returns
        -------
        unique_values : list[T]
            List of unique values in the column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3, 2, 4, 3])
        >>> column.get_unique_values()
        [1, 2, 3, 4]
        """
        return list(self._data.unique())

    def get_value(self, index: int) -> T:
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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3])
        >>> column.get_value(1)
        2
        """
        if index < 0 or index >= self._data.size:
            raise IndexOutOfBoundsError(index)

        return self._data[index]

    # ------------------------------------------------------------------------------------------------------------------
    # Information
    # ------------------------------------------------------------------------------------------------------------------

    def all(self, predicate: Callable[[T], bool]) -> bool:
        """
        Check if all values have a given property.

        Parameters
        ----------
        predicate : Callable[[T], bool])
            Callable that is used to find matches.

        Returns
        -------
        result : bool
            True if all match.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3])
        >>> column.all(lambda x: x < 4)
        True

        >>> column.all(lambda x: x < 2)
        False
        """
        return all(predicate(value) for value in self._data)

    def any(self, predicate: Callable[[T], bool]) -> bool:
        """
        Check if any value has a given property.

        Parameters
        ----------
        predicate : Callable[[T], bool])
            Callable that is used to find matches.

        Returns
        -------
        result : bool
            True if any match.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3])
        >>> column.any(lambda x: x < 2)
        True

        >>> column.any(lambda x: x < 1)
        False
        """
        return any(predicate(value) for value in self._data)

    def none(self, predicate: Callable[[T], bool]) -> bool:
        """
        Check if no values has a given property.

        Parameters
        ----------
        predicate : Callable[[T], bool])
            Callable that is used to find matches.

        Returns
        -------
        result : bool
            True if none match.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("test", [1, 2, 3])
        >>> column1.none(lambda x: x < 1)
        True

        >>> column2 = Column("test", [1, 2, 3])
        >>> column2.none(lambda x: x > 1)
        False
        """
        return all(not predicate(value) for value in self._data)

    def has_missing_values(self) -> bool:
        """
        Return whether the column has missing values.

        Returns
        -------
        missing_values_exist : bool
            True if missing values exist.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("test", [1, 2, 3, None])
        >>> column1.has_missing_values()
        True

        >>> column2 = Column("test", [1, 2, 3])
        >>> column2.has_missing_values()
        False
        """
        return self.any(lambda value: value is None or (isinstance(value, Number) and np.isnan(value)))

    # ------------------------------------------------------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------------------------------------------------------

    def rename(self, new_name: str) -> Column:
        """
        Return a new column with a new name.

        The original column is not modified.

        Parameters
        ----------
        new_name : str
            The new name of the column.

        Returns
        -------
        column : Column
            A new column with the new name.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3])
        >>> column.rename("new name")
        Column('new name', [1, 2, 3])
        """
        return Column._from_pandas_series(self._data.rename(new_name), self._type)

    def transform(self, transformer: Callable[[T], R]) -> Column[R]:
        """
        Apply a transform method to every data point.

        The original column is not modified.

        Parameters
        ----------
        transformer : Callable[[T], R]
            Function that will be applied to all data points.

        Returns
        -------
        transformed_column: Column
            The transformed column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> price = Column("price", [4.99, 5.99, 2.49])
        >>> sale = price.transform(lambda amount: amount * 0.8)
        """
        return Column(self.name, self._data.apply(transformer, convert_dtype=True))

    # ------------------------------------------------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------------------------------------------------

    def correlation_with(self, other_column: Column) -> float:
        """
        Calculate Pearson correlation between this and another column. Both columns have to be numerical.

        Returns
        -------
        correlation : float
            Correlation between the two columns.

        Raises
        ------
        NonNumericColumnError
            If one of the columns is not numerical.
        ColumnLengthMismatchError
            If the columns have different lengths.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("test", [1, 2, 3])
        >>> column2 = Column("test", [2, 4, 6])
        >>> column1.correlation_with(column2)
        1.0

        >>> column1 = Column("test", [1, 2, 3])
        >>> column2 = Column("test", [0.5, 4, -6])
        >>> column1.correlation_with(column2)
        -0.6404640308067906
        """
        if not self._type.is_numeric() or not other_column._type.is_numeric():
            raise NonNumericColumnError(
                f"Columns must be numerical. {self.name} is {self._type}, {other_column.name} is {other_column._type}.",
            )
        if self._data.size != other_column._data.size:
            raise ColumnLengthMismatchError(
                f"{self.name} is of size {self._data.size}, {other_column.name} is of size {other_column._data.size}.",
            )
        return self._data.corr(other_column._data)

    def idness(self) -> float:
        r"""
        Calculate the idness of this column.

        We define the idness as follows:

        $$
        \frac{\text{number of different values}}{\text{number of rows}}
        $$

        Returns
        -------
        idness : float
            The idness of the column.

        Raises
        ------
        ColumnSizeError
            If this column is empty.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("test", [1, 2, 3])
        >>> column1.idness()
        1.0

        >>> column2 = Column("test", [1, 2, 3, 2])
        >>> column2.idness()
        0.75
        """
        if self._data.size == 0:
            raise ColumnSizeError("> 0", "0")
        return self._data.nunique() / self._data.size

    def maximum(self) -> float:
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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3])
        >>> column.maximum()
        3
        """
        if not self._type.is_numeric():
            raise NonNumericColumnError(f"{self.name} is of type {self._type}.")
        return self._data.max()

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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3])
        >>> column.mean()
        2.0
        """
        if not self._type.is_numeric():
            raise NonNumericColumnError(f"{self.name} is of type {self._type}.")
        return self._data.mean()

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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3, 4])
        >>> column.median()
        2.5

        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3, 4, 5])
        >>> column.median()
        3.0
        """
        if not self._type.is_numeric():
            raise NonNumericColumnError(f"{self.name} is of type {self._type}.")
        return self._data.median()

    def minimum(self) -> float:
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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3, 4])
        >>> column.minimum()
        1
        """
        if not self._type.is_numeric():
            raise NonNumericColumnError(f"{self.name} is of type {self._type}.")
        return self._data.min()

    def missing_value_ratio(self) -> float:
        """
        Return the ratio of missing values to the total number of elements in the column.

        Returns
        -------
        ratio : float
            The ratio of missing values to the total number of elements in the column.

        Raises
        ------
        ColumnSizeError
            If the column is empty.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("test", [1, 2, 3, 4])
        >>> column1.missing_value_ratio()
        0.0

        >>> column2 = Column("test", [1, 2, 3, None])
        >>> column2.missing_value_ratio()
        0.25
        """
        if self._data.size == 0:
            raise ColumnSizeError("> 0", "0")
        return self._count_missing_values() / self._data.size

    def mode(self) -> list[T]:
        """
        Return the mode of the column.

        Returns
        -------
        mode: list[T]
            Returns a list with the most common values.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("test", [1, 2, 3, 3, 4])
        >>> column1.mode()
        [3]

        >>> column2 = Column("test", [1, 2, 3, 3, 4, 4])
        >>> column2.mode()
        [3, 4]
        """
        return self._data.mode().tolist()

    def stability(self) -> float:
        r"""
        Calculate the stability of this column.

        We define the stability as follows:

        $$
        \frac{\text{number of occurrences of most common non-null value}}{\text{number of non-null values}}
        $$

        The stability is not definded for a column with only null values.

        Returns
        -------
        stability : float
            The stability of the column.

        Raises
        ------
        ColumnSizeError
            If the column is empty.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("test", [1, 1, 2, 3])
        >>> column1.stability()
        0.5

        >>> column2 = Column("test", [1, 2, 2, 2, 3])
        >>> column2.stability()
        0.6
        """
        if self._data.size == 0:
            raise ColumnSizeError("> 0", "0")

        if self.all(lambda x: x is None):
            raise ValueError("Stability is not definded for a column with only null values.")

        return self._data.value_counts()[self.mode()[0]] / self._data.count()

    def standard_deviation(self) -> float:
        """
        Return the standard deviation of the column. The column has to be numerical.

        Returns
        -------
        sum : float
            The standard deviation of all values.

        Raises
        ------
        NonNumericColumnError
            If the data contains non-numerical data.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("test", [1, 2, 3])
        >>> column1.standard_deviation()
        1.0

        >>> column2 = Column("test", [1, 2, 4, 8, 16])
        >>> column2.standard_deviation()
        6.099180272790763
        """
        if not self.type.is_numeric():
            raise NonNumericColumnError(f"{self.name} is of type {self._type}.")
        return self._data.std()

    def sum(self) -> float:
        """
        Return the sum of the column. The column has to be numerical.

        Returns
        -------
        sum : float
            The sum of all values.

        Raises
        ------
        NonNumericColumnError
            If the data contains non-numerical data.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3])
        >>> column.sum()
        6
        """
        if not self.type.is_numeric():
            raise NonNumericColumnError(f"{self.name} is of type {self._type}.")
        return self._data.sum()

    def variance(self) -> float:
        """
        Return the variance of the column. The column has to be numerical.

        Returns
        -------
        sum : float
            The variance of all values.

        Raises
        ------
        NonNumericColumnError
            If the data contains non-numerical data.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3, 4, 5])
        >>> column.variance()
        2.5
        """
        if not self.type.is_numeric():
            raise NonNumericColumnError(f"{self.name} is of type {self._type}.")

        return self._data.var()

    # ------------------------------------------------------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------------------------------------------------------

    def plot_boxplot(self) -> Image:
        """
        Plot this column in a boxplot. This function can only plot real numerical data.

        Returns
        -------
        plot: Image
            The plot as an image.

        Raises
        ------
        NonNumericColumnError
            If the data contains non-numerical data.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3])
        >>> boxplot = column.plot_boxplot()
        """
        if not self.type.is_numeric():
            raise NonNumericColumnError(f"{self.name} is of type {self._type}.")

        fig = plt.figure()
        ax = sns.boxplot(data=self._data)
        ax.set(title=self.name)
        ax.set_xticks([])
        plt.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        plt.close()  # Prevents the figure from being displayed directly
        buffer.seek(0)
        return Image(buffer, ImageFormat.PNG)

    def plot_histogram(self) -> Image:
        """
        Plot a column in a histogram.

        Returns
        -------
        plot: Image
            The plot as an image.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3])
        >>> histogram = column.plot_histogram()
        """
        fig = plt.figure()
        ax = sns.histplot(data=self._data)
        ax.set_xticks(ax.get_xticks())
        ax.set(xlabel=self.name)
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
        return Image(buffer, ImageFormat.PNG)

    # ------------------------------------------------------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------------------------------------------------------

    def to_html(self) -> str:
        r"""
        Return an HTML representation of the column.

        Returns
        -------
        output : str
            The generated HTML.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3])
        >>> column.to_html()
        '<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th></th>\n      <th>test</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>2</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>3</td>\n    </tr>\n  </tbody>\n</table>'
        """
        frame = self._data.to_frame()
        frame.columns = [self.name]

        return frame.to_html(max_rows=self._data.size, max_cols=1)

    # ------------------------------------------------------------------------------------------------------------------
    # IPython integration
    # ------------------------------------------------------------------------------------------------------------------

    def _repr_html_(self) -> str:
        r"""
        Return an HTML representation of the column.

        Returns
        -------
        output : str
            The generated HTML.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("col_1", ['a', 'b', 'c'])
        >>> column._repr_html_()
        '<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th></th>\n      <th>col_1</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>a</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>b</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>c</td>\n    </tr>\n  </tbody>\n</table>\n</div>'
        """
        frame = self._data.to_frame()
        frame.columns = [self.name]

        return frame.to_html(max_rows=self._data.size, max_cols=1, notebook=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Other
    # ------------------------------------------------------------------------------------------------------------------

    def _count_missing_values(self) -> int:
        """
        Return the number of null values in the column.

        Returns
        -------
        count : int
            The number of null values.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("col_1", [None, 'a', None])
        >>> column._count_missing_values()
        2
        """
        return self._data.isna().sum()

    # ------------------------------------------------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------------------------------------------------

    def _copy(self) -> Column:
        """
        Return a copy of this column.

        Returns
        -------
        column : Column
            The copy of this column.

        """
        return copy.deepcopy(self)
