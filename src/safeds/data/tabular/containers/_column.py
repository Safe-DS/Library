from __future__ import annotations

import io
import sys
from collections.abc import Sequence
from numbers import Number
from typing import TYPE_CHECKING, Any, TypeVar, overload

from safeds._utils import _structural_hash
from safeds.data.image.containers import Image
from safeds.data.tabular.typing import ColumnType
from safeds.exceptions import (
    ColumnLengthMismatchError,
    ColumnSizeError,
    IndexOutOfBoundsError,
    NonNumericColumnError,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import pandas as pd

    from safeds.data.tabular.containers import Table

T = TypeVar("T")
R = TypeVar("R")


class Column(Sequence[T]):
    """
    A column is a named collection of values.

    Parameters
    ----------
    name:
        The name of the column.
    data:
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
        data:
            The data.
        type_:
            The type. If None, the type is inferred from the data.

        Returns
        -------
        column:
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
        name:
            The name of the column.
        data:
            The data. If None, an empty column is created.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3])
        """
        import pandas as pd

        # Enable copy-on-write for pandas dataframes
        pd.options.mode.copy_on_write = True

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
        other:
            The other object.

        Returns
        -------
        equal:
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
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> Column[T]: ...

    def __getitem__(self, index: int | slice) -> T | Column[T]:
        """
        Return the value of the specified row or rows.

        Parameters
        ----------
        index:
            The index of the row, or a slice specifying the start and end index.

        Returns
        -------
        value:
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

    def __hash__(self) -> int:
        """
        Return a deterministic hash value for this column.

        Returns
        -------
        hash:
            The hash value.
        """
        return _structural_hash(self.name, self.type.__repr__(), self.number_of_rows)

    def __iter__(self) -> Iterator[T]:
        r"""
        Create an iterator for the data of this column. This way e.g. for-each loops can be used on it.

        Returns
        -------
        iterator:
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
        n_rows:
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
        representation:
            The string representation.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3])
        >>> repr(column)
        "Column('test', [1, 2, 3])"
        """
        return f"Column({self._name!r}, {list(self._data)!r})"

    def __sizeof__(self) -> int:
        """
        Return the complete size of this object.

        Returns
        -------
        size:
            Size of this object in bytes.
        """
        return sys.getsizeof(self._data) + sys.getsizeof(self._name) + sys.getsizeof(self._type)

    def __str__(self) -> str:
        """
        Return a user-friendly string representation of this column.

        Returns
        -------
        representation:
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
        name:
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
        number_of_rows:
            The number of elements.
        """
        return len(self._data)

    @property
    def type(self) -> ColumnType:
        """
        Return the type of the column.

        Returns
        -------
        type:
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
        unique_values:
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
        index:
            Index of requested element.

        Returns
        -------
        value:
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
        predicate:
            Callable that is used to find matches.

        Returns
        -------
        result:
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
        predicate:
            Callable that is used to find matches.

        Returns
        -------
        result:
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
        predicate:
            Callable that is used to find matches.

        Returns
        -------
        result:
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
        missing_values_exist:
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
        import numpy as np

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
        new_name:
            The new name of the column.

        Returns
        -------
        column:
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
        transformer:
            Function that will be applied to all data points.

        Returns
        -------
        transformed_column:
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

    def summarize_statistics(self) -> Table:
        """
        Return a table with a number of statistical key values.

        The original Column is not modified.

        Returns
        -------
        statistics:
            The table with statistics.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 3])
        >>> column.summarize_statistics()
                         metric                   a
        0               minimum                   1
        1               maximum                   3
        2                  mean                 2.0
        3                  mode              [1, 3]
        4                median                 2.0
        5              variance                 2.0
        6    standard deviation  1.4142135623730951
        7   missing value count                   0
        8   missing value ratio                 0.0
        9                idness                 1.0
        10            stability                 0.5
        """
        from safeds.data.tabular.containers import Table

        return Table({self._name: self._data}).summarize_statistics()

    def correlation_with(self, other_column: Column) -> float:
        """
        Calculate Pearson correlation between this and another column. Both columns have to be numerical.

        Returns
        -------
        correlation:
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
        idness:
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
        max:
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
        mean:
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
        median:
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
        min:
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

    def missing_value_count(self) -> int:
        """
        Return the number of missing values in the column.

        Returns
        -------
        count:
            The number of missing values.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("col_1", [None, 'a', None])
        >>> column.missing_value_count()
        2
        """
        return self._data.isna().sum()

    def missing_value_ratio(self) -> float:
        """
        Return the ratio of missing values to the total number of elements in the column.

        Returns
        -------
        ratio:
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
        return self.missing_value_count() / self._data.size

    def mode(self) -> list[T]:
        """
        Return the mode of the column.

        Returns
        -------
        mode:
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
        stability:
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
        sum:
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
        sum:
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
        sum:
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
        plot:
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
        import matplotlib.pyplot as plt
        import seaborn as sns

        if not self.type.is_numeric():
            raise NonNumericColumnError(f"{self.name} is of type {self._type}.")

        fig = plt.figure()
        ax = sns.boxplot(data=self._data)
        ax.set(title=self.name)
        ax.set_xticks([])
        ax.set_ylabel("")
        plt.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        plt.close()  # Prevents the figure from being displayed directly
        buffer.seek(0)
        return Image.from_bytes(buffer.read())

    def plot_histogram(self, *, number_of_bins: int = 10) -> Image:
        """
        Plot a column in a histogram.

        Parameters
        ----------
        number_of_bins:
            The number of bins to use in the histogram. Default is 10.

        Returns
        -------
        plot:
            The plot as an image.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3])
        >>> histogram = column.plot_histogram()
        """
        from safeds.data.tabular.containers import Table

        return Table({self._name: self._data}).plot_histograms(number_of_bins=number_of_bins)

    def plot_compare_columns(self, column_list: list[Column]) -> Image:
        """
        Create a plot comparing the numerical values of columns using IDs as the x-axis.

        Parameters
        ----------
        column_list:
            A list of time columns to be plotted.

        Returns
        -------
        plot:
            A plot with all the Columns plotted by the ID on the x-axis.

        Raises
        ------
        NonNumericColumnError
            if the target column contains non numerical values
        ValueError
            if the columns do not have the same size

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> col1 =Column("target", [4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        >>> col2 =Column("target", [42, 51, 63, 71, 83, 91, 10, 11, 12, 13])
        >>> image = col1.plot_compare_columns([col2])
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns

        data = pd.DataFrame()
        column_list.append(self)
        size = len(column_list[0])
        data["INDEX"] = pd.DataFrame({"INDEX": range(size)})
        for index, col in enumerate(column_list):
            if not col.type.is_numeric():
                raise NonNumericColumnError("The time series plotted column contains non-numerical columns.")
            if len(col) != size:
                raise ValueError("The columns must have the same size.")
            data[col.name + " " + str(index)] = col._data

        fig = plt.figure()
        data = pd.melt(data, ["INDEX"])
        sns.lineplot(x="INDEX", y="value", hue="variable", data=data)
        plt.title("Multiple Series Plot")
        plt.xlabel("Time")

        plt.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        plt.close()  # Prevents the figure from being displayed directly
        buffer.seek(0)
        return Image.from_bytes(buffer.read())

    def plot_lagplot(self, lag: int) -> Image:
        """
        Plot a lagplot for the given column.

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
            If the column contains non-numerical values.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> table = Column("values", [1,2,3,4,3,2])
        >>> image = table.plot_lagplot(2)
        """
        import matplotlib.pyplot as plt
        import pandas as pd

        if not self.type.is_numeric():
            raise NonNumericColumnError("This time series target contains non-numerical columns.")
        ax = pd.plotting.lag_plot(self._data, lag=lag)
        fig = ax.figure
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        plt.close()  # Prevents the figure from being displayed directly
        buffer.seek(0)
        return Image.from_bytes(buffer.read())

    # ------------------------------------------------------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------------------------------------------------------

    def to_table(self) -> Table:
        """
        Create a table that contains only this column.

        Returns
        -------
        table:
            The table with this column.
        """
        # Must be imported here to avoid circular imports
        from safeds.data.tabular.containers import Table

        return Table.from_columns([self])

    def to_html(self) -> str:
        """
        Return an HTML representation of the column.

        Returns
        -------
        output:
            The generated HTML.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3])
        >>> html = column.to_html()
        """
        frame = self._data.to_frame()
        frame.columns = [self.name]

        return frame.to_html(max_rows=self._data.size, max_cols=1)

    # ------------------------------------------------------------------------------------------------------------------
    # IPython integration
    # ------------------------------------------------------------------------------------------------------------------

    def _repr_html_(self) -> str:
        """
        Return an HTML representation of the column.

        Returns
        -------
        output:
            The generated HTML.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("col_1", ['a', 'b', 'c'])
        >>> html = column._repr_html_()
        """
        frame = self._data.to_frame()
        frame.columns = [self.name]

        return frame.to_html(max_rows=self._data.size, max_cols=1, notebook=True)
