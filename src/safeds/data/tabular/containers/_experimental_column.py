from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, overload

from safeds._utils import _structural_hash
from safeds.data.tabular.plotting._experimental_column_plotter import ExperimentalColumnPlotter
from safeds.data.tabular.typing._experimental_polars_data_type import _PolarsDataType
from safeds.exceptions import IndexOutOfBoundsError

from ._column import Column
from ._experimental_vectorized_cell import _VectorizedCell

if TYPE_CHECKING:
    from polars import Series

    from safeds.data.tabular.typing._experimental_data_type import ExperimentalDataType

    from ._experimental_cell import ExperimentalCell
    from ._experimental_table import ExperimentalTable


T = TypeVar("T")
R = TypeVar("R")


class ExperimentalColumn(Sequence[T]):
    """
    A named, one-dimensional collection of homogeneous values.

    Parameters
    ----------
    name:
        The name of the column.
    data:
        The data of the column. If None, an empty column is created.

    Examples
    --------
    >>> from safeds.data.tabular.containers import ExperimentalColumn
    >>> ExperimentalColumn("test", [1, 2, 3])
    +------+
    | test |
    |  --- |
    |  i64 |
    +======+
    |    1 |
    |    2 |
    |    3 |
    +------+
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _from_polars_series(data: Series) -> ExperimentalColumn:
        result = object.__new__(ExperimentalColumn)
        result._series = data
        return result

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, name: str, data: Sequence[T] | None = None) -> None:
        import polars as pl

        if data is None:
            data = []

        self._series: pl.Series = pl.Series(name, data)

    def __contains__(self, item: Any) -> bool:
        return self._series.__contains__(item)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExperimentalColumn):
            return NotImplemented
        if self is other:
            return True
        return self._series.equals(other._series)

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> ExperimentalColumn[T]: ...

    def __getitem__(self, index: int | slice) -> T | ExperimentalColumn[T]:
        return self._series.__getitem__(index)

    def __hash__(self) -> int:
        return _structural_hash(
            self.name,
            self.type.__repr__(),
            self.number_of_rows,
        )

    def __iter__(self) -> Iterator[T]:
        return self._series.__iter__()

    def __len__(self) -> int:
        return self.number_of_rows

    def __repr__(self) -> str:
        return self.to_table().__repr__()

    def __sizeof__(self) -> int:
        return self._series.estimated_size()

    def __str__(self) -> str:
        return self.to_table().__str__()

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def is_numeric(self) -> bool:
        """Whether the column is numeric."""
        return self._series.dtype.is_numeric()

    @property
    def is_temporal(self) -> bool:
        """Whether the column is temporal."""
        return self._series.dtype.is_temporal()

    @property
    def name(self) -> str:
        """The name of the column."""
        return self._series.name

    @property
    def number_of_rows(self) -> int:
        """The number of rows in the column."""
        return self._series.len()

    @property
    def plot(self) -> ExperimentalColumnPlotter:
        """The plotter for the column."""
        return ExperimentalColumnPlotter(self)

    @property
    def type(self) -> ExperimentalDataType:
        """The type of the column."""
        return _PolarsDataType(self._series.dtype)

    # ------------------------------------------------------------------------------------------------------------------
    # Value operations
    # ------------------------------------------------------------------------------------------------------------------

    def get_distinct_values(self) -> list[T]:
        """
        Return the distinct values in the column.

        Returns
        -------
        distinct_values:
            The distinct values in the column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalColumn
        >>> column = ExperimentalColumn("test", [1, 2, 3, 2])
        >>> column.get_distinct_values()
        [1, 2, 3]
        """
        return self._series.unique().sort().to_list()

    def get_value(self, index: int) -> T:
        """
        Return the column value at specified index. Indexing starts at 0.

        Parameters
        ----------
        index:
            Index of requested value.

        Returns
        -------
        value:
            Value at index.

        Raises
        ------
        IndexError
            If the given index does not exist in the column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalColumn
        >>> column = ExperimentalColumn("test", [1, 2, 3])
        >>> column.get_value(1)
        2
        """
        if index < 0 or index >= self.number_of_rows:
            raise IndexOutOfBoundsError(index)

        return self._series[index]

    # ------------------------------------------------------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------------------------------------------------------

    def all(self, predicate: Callable[[ExperimentalCell[T]], ExperimentalCell[bool]]) -> bool:
        """
        Return whether all values in the column satisfy the predicate.

        Parameters
        ----------
        predicate:
            The predicate to apply to each value.

        Returns
        -------
        all_satisfy_predicate:
            Whether all values in the column satisfy the predicate.

        Raises
        ------
        TypeError
            If the predicate does not return a boolean cell.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalColumn
        >>> column = ExperimentalColumn("test", [1, 2, 3])
        >>> column.all(lambda cell: cell > 0)
        True

        >>> column.all(lambda cell: cell < 3)
        False
        """
        import polars as pl

        result = predicate(_VectorizedCell(self))
        if not isinstance(result, _VectorizedCell) or not result._series.dtype.is_(pl.Boolean):
            raise TypeError("The predicate must return a boolean cell.")

        return result._series.all()

    def any(self, predicate: Callable[[ExperimentalCell[T]], ExperimentalCell[bool]]) -> bool:
        """
        Return whether any value in the column satisfies the predicate.

        Parameters
        ----------
        predicate:
            The predicate to apply to each value.

        Returns
        -------
        any_satisfy_predicate:
            Whether any value in the column satisfies the predicate.

        Raises
        ------
        TypeError
            If the predicate does not return a boolean cell.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalColumn
        >>> column = ExperimentalColumn("test", [1, 2, 3])
        >>> column.any(lambda cell: cell > 2)
        True

        >>> column.any(lambda cell: cell < 0)
        False
        """
        import polars as pl

        result = predicate(_VectorizedCell(self))
        if not isinstance(result, _VectorizedCell) or not result._series.dtype.is_(pl.Boolean):
            raise TypeError("The predicate must return a boolean cell.")

        return result._series.any()

    def count(self, predicate: Callable[[ExperimentalCell[T]], ExperimentalCell[bool]]) -> int:
        """
        Return how many values in the column satisfy the predicate.

        Parameters
        ----------
        predicate:
            The predicate to apply to each value.

        Returns
        -------
        count:
            The number of values in the column that satisfy the predicate.

        Raises
        ------
        TypeError
            If the predicate does not return a boolean cell.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalColumn
        >>> column = ExperimentalColumn("test", [1, 2, 3])
        >>> column.count(lambda cell: cell > 1)
        2

        >>> column.count(lambda cell: cell < 0)
        0
        """
        import polars as pl

        result = predicate(_VectorizedCell(self))
        if not isinstance(result, _VectorizedCell) or not result._series.dtype.is_(pl.Boolean):
            raise TypeError("The predicate must return a boolean cell.")

        return result._series.sum()

    def none(self, predicate: Callable[[ExperimentalCell[T]], ExperimentalCell[bool]]) -> bool:
        """
        Return whether no value in the column satisfies the predicate.

        Parameters
        ----------
        predicate:
            The predicate to apply to each value.

        Returns
        -------
        none_satisfy_predicate:
            Whether no value in the column satisfies the predicate.

        Raises
        ------
        TypeError
            If the predicate does not return a boolean cell.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalColumn
        >>> column = ExperimentalColumn("test", [1, 2, 3])
        >>> column.none(lambda cell: cell < 0)
        True

        >>> column.none(lambda cell: cell > 2)
        False
        """
        import polars as pl

        result = predicate(_VectorizedCell(self))
        if not isinstance(result, _VectorizedCell) or not result._series.dtype.is_(pl.Boolean):
            raise TypeError("The predicate must return a boolean cell.")

        return (~result._series).all()

    # ------------------------------------------------------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------------------------------------------------------

    def rename(self, new_name: str) -> ExperimentalColumn[T]:
        """
        Return a new column with a new name.

        The original column is not modified.

        Parameters
        ----------
        new_name:
            The new name of the column.

        Returns
        -------
        renamed_column:
            A new column with the new name.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalColumn
        >>> column = ExperimentalColumn("test", [1, 2, 3])
        >>> column.rename("new_name")
        +----------+
        | new_name |
        |      --- |
        |      i64 |
        +==========+
        |        1 |
        |        2 |
        |        3 |
        +----------+
        """
        return self._from_polars_series(self._series.rename(new_name))

    def transform(
        self,
        transformer: Callable[[ExperimentalCell[T]], ExperimentalCell[R]],
    ) -> ExperimentalColumn[R]:
        """
        Return a new column with values transformed by the transformer.

        The original column is not modified.

        Parameters
        ----------
        transformer:
            The transformer to apply to each value.

        Returns
        -------
        transformed_column:
            A new column with transformed values.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalColumn
        >>> column = ExperimentalColumn("test", [1, 2, 3])
        >>> column.transform(lambda cell: 2 * cell)
        +------+
        | test |
        |  --- |
        |  i64 |
        +======+
        |    2 |
        |    4 |
        |    6 |
        +------+
        """
        result = transformer(_VectorizedCell(self))
        if not isinstance(result, _VectorizedCell):
            raise TypeError("The transformer must return a cell.")

        return self._from_polars_series(result._series)

    # ------------------------------------------------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------------------------------------------------

    def summarize_statistics(self) -> ExperimentalTable:
        """
        Return a table with important statistics about the column.

        Returns
        -------
        statistics:
            The table with statistics.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalColumn
        >>> column = ExperimentalColumn("a", [1, 3])
        >>> column.summarize_statistics()
        +----------------------+--------------------+
        | metric               | a                  |
        | ---                  | ---                |
        | str                  | str                |
        +===========================================+
        | min                  | 1                  |
        | max                  | 3                  |
        | mean                 | 2.0                |
        | median               | 2.0                |
        | standard deviation   | 1.4142135623730951 |
        | distinct value count | 2                  |
        | idness               | 1.0                |
        | missing value ratio  | 0.0                |
        | stability            | 0.5                |
        +----------------------+--------------------+
        """
        from ._experimental_table import ExperimentalTable

        # TODO: turn this around (call table method, implement in table; allows parallelization)
        mean = self.mean() or "-"
        median = self.median() or "-"
        standard_deviation = self.standard_deviation() or "-"

        return ExperimentalTable(
            {
                "metric": [
                    "min",
                    "max",
                    "mean",
                    "median",
                    "standard deviation",
                    "distinct value count",
                    "idness",
                    "missing value ratio",
                    "stability",
                ],
                self.name: [
                    str(self.min()),
                    str(self.max()),
                    str(mean),
                    str(median),
                    str(standard_deviation),
                    str(self.distinct_value_count()),
                    str(self.idness()),
                    str(self.missing_value_ratio()),
                    str(self.stability()),
                ],
            },
        )

    def correlation_with(self, other: ExperimentalColumn) -> float:
        """
        Calculate the Pearson correlation between this column and another column.

        The Pearson correlation is a value between -1 and 1 that indicates how much the two columns are linearly related:
        * A correlation of -1 indicates a perfect negative linear relationship.
        * A correlation of 0 indicates no linear relationship.
        * A correlation of 1 indicates a perfect positive linear relationship.

        Parameters
        ----------
        other:
            The other column to calculate the correlation with.

        Returns
        -------
        correlation:
            The Pearson correlation between the two columns.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalColumn
        >>> column1 = ExperimentalColumn("test", [1, 2, 3])
        >>> column2 = ExperimentalColumn("test", [2, 4, 6])
        >>> column1.correlation_with(column2)
        1.0

        >>> column4 = ExperimentalColumn("test", [3, 2, 1])
        >>> column1.correlation_with(column4)
        -1.0
        """
        import polars as pl

        return pl.DataFrame({"a": self._series, "b": other._series}).corr().item(row=1, column="a")

    def distinct_value_count(self) -> int:
        """
        Return the number of distinct values in the column.

        Returns
        -------
        distinct_value_count:
            The number of distinct values in the column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalColumn
        >>> column = ExperimentalColumn("test", [1, 2, 3, 2])
        >>> column.distinct_value_count()
        3
        """
        return self._series.n_unique()

    def idness(self) -> float:
        """
        Calculate the idness of this column.

        We define the idness as the number of distinct values divided by the number of rows. If the column is empty,
        the idness is 1.0.

        A high idness indicates that the column most values in the column are unique. In this case, you must be careful
        when using the column for analysis, as a model may learn a mapping from this column to the target.

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
        >>> from safeds.data.tabular.containers import ExperimentalColumn
        >>> column1 = ExperimentalColumn("test", [1, 2, 3])
        >>> column1.idness()
        1.0

        >>> column2 = ExperimentalColumn("test", [1, 2, 3, 2])
        >>> column2.idness()
        0.75
        """
        if self.number_of_rows == 0:
            return 1.0

        return self.distinct_value_count() / self.number_of_rows

    def max(self) -> T:
        """
        Return the maximum value in the column.

        Returns
        -------
        max:
            The maximum value in the column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalColumn
        >>> column = ExperimentalColumn("test", [1, 2, 3])
        >>> column.max()
        3
        """
        return self._series.max()

    def mean(self) -> T:
        """
        Return the mean of the values in the column.

        The mean is the sum of the values divided by the number of values.

        Returns
        -------
        mean:
            The mean of the values in the column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalColumn
        >>> column = ExperimentalColumn("test", [1, 2, 3])
        >>> column.mean()
        2.0
        """
        return self._series.mean()

    def median(self) -> T:
        """
        Return the median of the values in the column.

        The median is the value in the middle of the sorted list of values. If the number of values is even, the median
        is the mean of the two middle values.

        Returns
        -------
        median:
            The median of the values in the column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalColumn
        >>> column = ExperimentalColumn("test", [1, 2, 3])
        >>> column.median()
        2.0
        """
        return self._series.median()

    def min(self) -> T:
        """
        Return the minimum value in the column.

        Returns
        -------
        min:
            The minimum value in the column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalColumn
        >>> column = ExperimentalColumn("test", [1, 2, 3])
        >>> column.min()
        1
        """
        return self._series.min()

    def missing_value_count(self) -> int:
        """
        Return the number of missing values in the column.

        Returns
        -------
        missing_value_count:
            The number of missing values in the column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalColumn
        >>> column = ExperimentalColumn("test", [1, None, 3])
        >>> column.missing_value_count()
        1
        """
        return self._series.null_count()

    def missing_value_ratio(self) -> float:
        """
        Return the missing value ratio.

        We define the missing value ratio as the number of missing values in the column divided by the number of rows.
        If the column is empty, the missing value ratio is 1.0.

        A high missing value ratio indicates that the column is dominated by missing values. In this case, the column
        may not be useful for analysis.

        Returns
        -------
        missing_value_ratio:
            The ratio of missing values in the column.
        """
        if self.number_of_rows == 0:
            return 1.0

        return self._series.null_count() / self.number_of_rows

    def mode(self) -> ExperimentalColumn[T]:
        """
        Return the mode of the values in the column.

        The mode is the value that appears most frequently in the column. If multiple values occur equally often, all
        of them are returned. The values are sorted in ascending order.

        Returns
        -------
        mode:
            The mode of the values in the column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalColumn
        >>> column = ExperimentalColumn("test", [3, 1, 2, 1, 3])
        >>> column.mode()
        +------+
        | test |
        |  --- |
        |  i64 |
        +======+
        |    1 |
        |    3 |
        +------+
        """
        return self._from_polars_series(self._series.mode().sort())

    def stability(self) -> float:
        """
        Return the stability of the column.

        We define the stability as the number of occurrences of the most common non-missing value divided by the total
        number of non-missing values. If the column is empty or all values are missing, the stability is 1.0.

        A high stability indicates that the column is dominated by a single value. In this case, the column may not be
        useful for analysis.

        Returns
        -------
        stability:
            The stability of the column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalColumn
        >>> column = ExperimentalColumn("test", [1, 1, 2, 3, None])
        >>> column.stability()
        0.5
        """
        non_missing = self._series.drop_nulls()
        if non_missing.len() == 0:
            return 1.0

        mode_count = non_missing.unique_counts().max()

        return mode_count / non_missing.len()

    def standard_deviation(self) -> float | None:
        """
        Return the standard deviation of the values in the column.

        The standard deviation is the square root of the variance.

        Returns
        -------
        standard_deviation:
            The standard deviation of the values in the column. If no standard deviation can be calculated due to the
            type of the column, None is returned.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalColumn
        >>> column = ExperimentalColumn("test", [1, 2, 3])
        >>> column.standard_deviation()
        1.0
        """
        from polars.exceptions import InvalidOperationError

        try:
            return self._series.std()
        except InvalidOperationError:
            return None

    def variance(self) -> float | None:
        """
        Return the variance of the values in the column.

        The variance is the average of the squared differences from the mean.

        Returns
        -------
        variance:
            The variance of the values in the column. If no variance can be calculated due to the type of the column,
            None is returned.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalColumn
        >>> column = ExperimentalColumn("test", [1, 2, 3])
        >>> column.variance()
        1.0
        """
        from polars.exceptions import InvalidOperationError

        try:
            return self._series.var()
        except InvalidOperationError:
            return None

    # ------------------------------------------------------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------------------------------------------------------

    def to_list(self) -> list[T]:
        """
        Return the values of the column in a list.

        Returns
        -------
        values:
            The values of the column in a list.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalColumn
        >>> column = ExperimentalColumn("test", [1, 2, 3])
        >>> column.to_list()
        [1, 2, 3]
        """
        return self._series.to_list()

    def to_table(self) -> ExperimentalTable:
        """
        Create a table that contains only this column.

        Returns
        -------
        table:
            The table with this column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalColumn
        >>> column = ExperimentalColumn("test", [1, 2, 3])
        >>> column.to_table()
        +------+
        | test |
        |  --- |
        |  i64 |
        +======+
        |    1 |
        |    2 |
        |    3 |
        +------+
        """
        from ._experimental_table import ExperimentalTable

        return ExperimentalTable._from_polars_data_frame(self._series.to_frame())

    def temporary_to_old_column(self) -> Column:
        """
        Convert the column to the old column format. This method is temporary and will be removed in a later version.

        Returns
        -------
        old_column:
            The column in the old format.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalColumn
        >>> column = ExperimentalColumn("a", [1, 2, 3])
        >>> old_column = column.temporary_to_old_column()
        """
        return Column._from_pandas_series(self._series.to_pandas())

    # ------------------------------------------------------------------------------------------------------------------
    # IPython integration
    # ------------------------------------------------------------------------------------------------------------------

    def _repr_html_(self) -> str:
        """
        Return a compact HTML representation of the column for IPython.

        Note that this operation must fully load the data into memory, which can be expensive.

        Returns
        -------
        html:
            The generated HTML.
        """
        return self._series._repr_html_()
