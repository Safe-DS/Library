from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Literal, TypeVar, overload

from safeds._utils import _safe_collect_lazy_frame, _safe_collect_lazy_frame_schema, _structural_hash
from safeds._validation import (
    _check_column_has_no_missing_values,
    _check_column_is_numeric,
    _check_indices,
    _check_row_counts_are_equal,
)
from safeds.data.tabular.plotting import ColumnPlotter
from safeds.data.tabular.typing._polars_column_type import _PolarsColumnType

from ._lazy_cell import _LazyCell

if TYPE_CHECKING:
    import polars as pl

    from safeds.data.tabular.typing import ColumnType
    from safeds.exceptions import (  # noqa: F401
        ColumnTypeError,
        IndexOutOfBoundsError,
        LengthMismatchError,
        MissingValuesError,
    )

    from ._cell import Cell
    from ._table import Table


T_co = TypeVar("T_co", covariant=True)
R_co = TypeVar("R_co", covariant=True)


class Column(Sequence[T_co]):
    """
    A named, one-dimensional collection of homogeneous values.

    Parameters
    ----------
    name:
        The name of the column.
    data:
        The data of the column.
    type:
        The type of the column. If `None` (default), the type is inferred from the data.

    Examples
    --------
    >>> from safeds.data.tabular.containers import Column
    >>> Column("a", [1, 2, 3])
    +-----+
    |   a |
    | --- |
    | i64 |
    +=====+
    |   1 |
    |   2 |
    |   3 |
    +-----+

    >>> from safeds.data.tabular.typing import ColumnType
    >>> Column("a", [1, 2, 3], type=ColumnType.string())
    +-----+
    | a   |
    | --- |
    | str |
    +=====+
    | 1   |
    | 2   |
    | 3   |
    +-----+
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _from_polars_lazy_frame(name: str, data: pl.LazyFrame) -> Column:
        result = object.__new__(Column)
        result._name = name
        result._lazy_frame = data.select(name)
        result.__series_cache = None
        return result

    @staticmethod
    def _from_polars_series(data: pl.Series) -> Column:
        result = object.__new__(Column)
        result._name = data.name
        result._lazy_frame = data.to_frame().lazy()
        result.__series_cache = data
        return result

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        name: str,
        data: Sequence[T_co],
        *,
        type: ColumnType | None = None,
    ) -> None:
        import polars as pl

        # Preprocessing
        dtype = None if type is None else type._polars_data_type

        # Implementation
        self._name: str = name
        self.__series_cache: pl.Series | None = pl.Series(name, data, dtype=dtype, strict=False)
        self._lazy_frame: pl.LazyFrame = self.__series_cache.to_frame().lazy()

    def __contains__(self, value: object) -> bool:
        import polars as pl

        try:
            return self._series.__contains__(value)
        except pl.InvalidOperationError:
            # Happens if types are incompatible
            return False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Column):
            return NotImplemented
        if self is other:
            return True
        return self.name == other.name and self._series.equals(other._series)

    @overload
    def __getitem__(self, index: int) -> T_co: ...

    @overload
    def __getitem__(self, index: slice) -> Column[T_co]: ...

    def __getitem__(self, index: int | slice) -> T_co | Column[T_co]:
        if isinstance(index, int):
            return self.get_value(index)

        try:
            return self._from_polars_lazy_frame(self.name, self._lazy_frame[index])
        except ValueError:
            return self._from_polars_series(self._series[index])

    def __hash__(self) -> int:
        return _structural_hash(
            self.name,
            self.type.__repr__(),
            self.row_count,
        )

    def __iter__(self) -> Iterator[T_co]:
        return self._series.__iter__()

    def __len__(self) -> int:
        return self.row_count

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
    def _series(self) -> pl.Series:
        if self.__series_cache is None:
            self.__series_cache = _safe_collect_lazy_frame(self._lazy_frame).to_series()

        return self.__series_cache

    @property
    def name(self) -> str:
        """
        The name of the column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, 3])
        >>> column.name
        'a'
        """
        return self._name

    @property
    def row_count(self) -> int:
        """
        The number of rows.

        **Note:** This operation must fully load the data into memory, which can be expensive.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, 3])
        >>> column.row_count
        3
        """
        return self._series.len()

    @property
    def plot(self) -> ColumnPlotter:
        """
        The plotter for the column.

        Call methods of the plotter to create various plots for the column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, 3])
        >>> plot = column.plot.box_plot()
        """
        return ColumnPlotter(self)

    @property
    def type(self) -> ColumnType:
        """
        The type of the column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, 3])
        >>> column.type
        int64
        """
        schema = _safe_collect_lazy_frame_schema(self._lazy_frame)
        return _PolarsColumnType(schema.dtypes()[0])

    # ------------------------------------------------------------------------------------------------------------------
    # Value operations
    # ------------------------------------------------------------------------------------------------------------------

    @overload
    def get_distinct_values(
        self,
        *,
        ignore_missing_values: Literal[True] = ...,
    ) -> Sequence[T_co]: ...

    @overload
    def get_distinct_values(
        self,
        *,
        ignore_missing_values: bool,
    ) -> Sequence[T_co | None]: ...

    def get_distinct_values(
        self,
        *,
        ignore_missing_values: bool = True,
    ) -> Sequence[T_co | None]:
        """
        Return the distinct values in the column.

        Parameters
        ----------
        ignore_missing_values:
            Whether to ignore missing values.

        Returns
        -------
        distinct_values:
            The distinct values in the column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, 3, 2])
        >>> column.get_distinct_values()
        [1, 2, 3]
        """
        import polars as pl

        if self.row_count == 0:
            return []  # polars raises otherwise
        elif self._series.dtype == pl.Null:
            # polars raises otherwise
            if ignore_missing_values:
                return []
            else:
                return [None]

        if ignore_missing_values:
            series = self._series.drop_nulls()
        else:
            series = self._series

        return series.unique(maintain_order=True).to_list()

    def get_value(self, index: int) -> T_co:
        """
        Return the column value at specified index. This is equivalent to the `[]` operator (indexed access).

        Nonnegative indices are counted from the beginning (starting at 0), negative indices from the end (starting at
        -1).

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
        IndexOutOfBoundsError
            If the index is out of bounds.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, 3])
        >>> column.get_value(0)
        1

        >>> column[0]
        1

        >>> column.get_value(-1)
        3

        >>> column[-1]
        3
        """
        _check_indices(self, index)

        # Lazy containers do not allow indexed accesses
        return self._series[index]

    # ------------------------------------------------------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------------------------------------------------------

    @overload
    def all(
        self,
        predicate: Callable[[Cell[T_co]], Cell[bool | None]],
        *,
        ignore_unknown: Literal[True] = ...,
    ) -> bool: ...

    @overload
    def all(
        self,
        predicate: Callable[[Cell[T_co]], Cell[bool | None]],
        *,
        ignore_unknown: bool,
    ) -> bool | None: ...

    def all(
        self,
        predicate: Callable[[Cell[T_co]], Cell[bool | None]],
        *,
        ignore_unknown: bool = True,
    ) -> bool | None:
        """
        Check whether all values in the column satisfy the predicate.

        The predicate can return one of three values:

        * True, if the value satisfies the predicate.
        * False, if the value does not satisfy the predicate.
        * None, if the truthiness of the predicate is unknown, e.g. due to missing values.

        By default, cases where the truthiness of the predicate is unknown are ignored and this method returns

        * True, if the predicate always returns True or None.
        * False, if the predicate returns False at least once.

        You can instead enable Kleene logic by setting `ignore_unknown=False`. In this case, this method returns

        * True, if the predicate always returns True.
        * False, if the predicate returns False at least once.
        * None, if the predicate never returns False, but at least once None.

        Parameters
        ----------
        predicate:
            The predicate to apply to each value.
        ignore_unknown:
            Whether to ignore cases where the truthiness of the predicate is unknown.

        Returns
        -------
        all_satisfy_predicate:
            Whether all values in the column satisfy the predicate.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, 3, None])
        >>> column.all(lambda cell: cell > 0)
        True

        >>> column.all(lambda cell: cell < 3)
        False

        >>> print(column.all(lambda cell: cell > 0, ignore_unknown=False))
        None

        >>> column.all(lambda cell: cell < 3, ignore_unknown=False)
        False
        """
        import polars as pl

        expression = predicate(_LazyCell(pl.col(self.name)))._polars_expression.all(ignore_nulls=ignore_unknown)
        frame = _safe_collect_lazy_frame(self._lazy_frame.select(expression))

        return frame.item()

    @overload
    def any(
        self,
        predicate: Callable[[Cell[T_co]], Cell[bool | None]],
        *,
        ignore_unknown: Literal[True] = ...,
    ) -> bool: ...

    @overload
    def any(
        self,
        predicate: Callable[[Cell[T_co]], Cell[bool | None]],
        *,
        ignore_unknown: bool,
    ) -> bool | None: ...

    def any(
        self,
        predicate: Callable[[Cell[T_co]], Cell[bool | None]],
        *,
        ignore_unknown: bool = True,
    ) -> bool | None:
        """
        Check whether any value in the column satisfies the predicate.

        The predicate can return one of three values:

        * True, if the value satisfies the predicate.
        * False, if the value does not satisfy the predicate.
        * None, if the truthiness of the predicate is unknown, e.g. due to missing values.

        By default, cases where the truthiness of the predicate is unknown are ignored and this method returns

        * True, if the predicate returns True at least once.
        * False, if the predicate always returns False or None.

        You can instead enable Kleene logic by setting `ignore_unknown=False`. In this case, this method returns

        * True, if the predicate returns True at least once.
        * False, if the predicate always returns False.
        * None, if the predicate never returns True, but at least once None.

        Parameters
        ----------
        predicate:
            The predicate to apply to each value.
        ignore_unknown:
            Whether to ignore cases where the truthiness of the predicate is unknown.

        Returns
        -------
        any_satisfy_predicate:
            Whether any value in the column satisfies the predicate.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, 3, None])
        >>> column.any(lambda cell: cell > 2)
        True

        >>> column.any(lambda cell: cell < 0)
        False

        >>> column.any(lambda cell: cell > 2, ignore_unknown=False)
        True

        >>> print(column.any(lambda cell: cell < 0, ignore_unknown=False))
        None
        """
        import polars as pl

        expression = predicate(_LazyCell(pl.col(self.name)))._polars_expression.any(ignore_nulls=ignore_unknown)
        frame = _safe_collect_lazy_frame(self._lazy_frame.select(expression))

        return frame.item()

    @overload
    def count_if(
        self,
        predicate: Callable[[Cell[T_co]], Cell[bool | None]],
        *,
        ignore_unknown: Literal[True] = ...,
    ) -> int: ...

    @overload
    def count_if(
        self,
        predicate: Callable[[Cell[T_co]], Cell[bool | None]],
        *,
        ignore_unknown: bool,
    ) -> int | None: ...

    def count_if(
        self,
        predicate: Callable[[Cell[T_co]], Cell[bool | None]],
        *,
        ignore_unknown: bool = True,
    ) -> int | None:
        """
        Count how many values in the column satisfy the predicate.

        The predicate can return one of three results:

        * True, if the value satisfies the predicate.
        * False, if the value does not satisfy the predicate.
        * None, if the truthiness of the predicate is unknown, e.g. due to missing values.

        By default, cases where the truthiness of the predicate is unknown are ignored and this method returns how
        often the predicate returns True.

        You can instead enable Kleene logic by setting `ignore_unknown=False`. In this case, this method returns None if
        the predicate returns None at least once. Otherwise, it still returns how often the predicate returns True.

        Parameters
        ----------
        predicate:
            The predicate to apply to each value.
        ignore_unknown:
            Whether to ignore cases where the truthiness of the predicate is unknown.

        Returns
        -------
        count:
            The number of values in the column that satisfy the predicate.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, 3, None])
        >>> column.count_if(lambda cell: cell > 1)
        2

        >>> print(column.count_if(lambda cell: cell < 0, ignore_unknown=False))
        None
        """
        import polars as pl

        expression = predicate(_LazyCell(pl.col(self.name)))._polars_expression
        frame = _safe_collect_lazy_frame(self._lazy_frame.select(expression))
        series = frame.to_series()

        if ignore_unknown or not series.has_nulls():
            return series.sum()
        else:
            return None

    @overload
    def none(
        self,
        predicate: Callable[[Cell[T_co]], Cell[bool | None]],
        *,
        ignore_unknown: Literal[True] = ...,
    ) -> bool: ...

    @overload
    def none(
        self,
        predicate: Callable[[Cell[T_co]], Cell[bool | None]],
        *,
        ignore_unknown: bool,
    ) -> bool | None: ...

    def none(
        self,
        predicate: Callable[[Cell[T_co]], Cell[bool | None]],
        *,
        ignore_unknown: bool = True,
    ) -> bool | None:
        """
        Check whether no value in the column satisfies the predicate.

        The predicate can return one of three values:

        * True, if the value satisfies the predicate.
        * False, if the value does not satisfy the predicate.
        * None, if the truthiness of the predicate is unknown, e.g. due to missing values.

        By default, cases where the truthiness of the predicate is unknown are ignored and this method returns

        * True, if the predicate always returns False or None.
        * False, if the predicate returns True at least once.

        You can instead enable Kleene logic by setting `ignore_unknown=False`. In this case, this method returns

        * True, if the predicate always returns False.
        * False, if the predicate returns True at least once.
        * None, if the predicate never returns True, but at least once None.

        Parameters
        ----------
        predicate:
            The predicate to apply to each value.
        ignore_unknown:
            Whether to ignore cases where the truthiness of the predicate is unknown.

        Returns
        -------
        none_satisfy_predicate:
            Whether no value in the column satisfies the predicate.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, 3, None])
        >>> column.none(lambda cell: cell < 0)
        True

        >>> column.none(lambda cell: cell > 2)
        False

        >>> print(column.none(lambda cell: cell < 0, ignore_unknown=False))
        None

        >>> column.none(lambda cell: cell > 2, ignore_unknown=False)
        False
        """
        import polars as pl

        expression = predicate(_LazyCell(pl.col(self.name)))._polars_expression.not_().all(ignore_nulls=ignore_unknown)
        frame = _safe_collect_lazy_frame(self._lazy_frame.select(expression))

        return frame.item()

    # ------------------------------------------------------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------------------------------------------------------

    def rename(self, new_name: str) -> Column[T_co]:
        """
        Rename the column and return the result as a new column.

        **Note:** The original column is not modified.

        Parameters
        ----------
        new_name:
            The new name of the column.

        Returns
        -------
        new_column:
            A column with the new name.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, 3])
        >>> column.rename("b")
        +-----+
        |   b |
        | --- |
        | i64 |
        +=====+
        |   1 |
        |   2 |
        |   3 |
        +-----+
        """
        result = self._lazy_frame.rename({self.name: new_name})
        return self._from_polars_lazy_frame(new_name, result)

    def transform(
        self,
        transformer: Callable[[Cell[T_co]], Cell[R_co]],
    ) -> Column[R_co]:
        """
        Transform the values in the column and return the result as a new column.

        **Note:** The original column is not modified.

        Parameters
        ----------
        transformer:
            The transformer to apply to each value.

        Returns
        -------
        new_column:
            A column with the transformed values.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, 3])
        >>> column.transform(lambda cell: 2 * cell)
        +-----+
        |   a |
        | --- |
        | i64 |
        +=====+
        |   2 |
        |   4 |
        |   6 |
        +-----+
        """
        import polars as pl

        expression = transformer(_LazyCell(pl.col(self.name)))._polars_expression.alias(self.name)
        result = self._lazy_frame.with_columns(expression)  # with_columns always keeps number of rows

        return self._from_polars_lazy_frame(self.name, result)

    # ------------------------------------------------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------------------------------------------------

    def summarize_statistics(self) -> Table:
        """
        Return a table with important statistics about the column.

        !!! warning "API Stability"

            Do not rely on the exact output of this method. In future versions, we may change the displayed statistics
            without prior notice.

        Returns
        -------
        statistics:
            The table with statistics.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 3])
        >>> column.summarize_statistics()
        +---------------------+---------+
        | statistic           |       a |
        | ---                 |     --- |
        | str                 |     f64 |
        +===============================+
        | min                 | 1.00000 |
        | max                 | 3.00000 |
        | mean                | 2.00000 |
        | median              | 2.00000 |
        | standard deviation  | 1.41421 |
        | missing value ratio | 0.00000 |
        | stability           | 0.50000 |
        | idness              | 1.00000 |
        +---------------------+---------+
        """
        return self.to_table().summarize_statistics()

    def correlation_with(self, other: Column) -> float:
        """
        Calculate the Pearson correlation between this column and another column.

        The Pearson correlation is a value between -1 and 1 that indicates how much the two columns are **linearly**
        related:

        - A correlation of -1 indicates a perfect negative linear relationship.
        - A correlation of 0 indicates no linear relationship.
        - A correlation of 1 indicates a perfect positive linear relationship.

        A value of 0 does not necessarily mean that the columns are independent. It only means that there is no linear
        relationship between the columns.

        Parameters
        ----------
        other:
            The other column to calculate the correlation with.

        Returns
        -------
        correlation:
            The Pearson correlation between the two columns.

        Raises
        ------
        ColumnTypeError
            If one of the columns is not numeric.
        LengthMismatchError
            If the columns have different lengths.
        ValueError
            If one of the columns has missing values.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("a", [1, 2, 3])
        >>> column2 = Column("a", [2, 4, 6])
        >>> column1.correlation_with(column2)
        1.0

        >>> column3 = Column("a", [3, 2, 1])
        >>> column1.correlation_with(column3)
        -1.0
        """
        import polars as pl

        _check_column_is_numeric(self, other_columns=[other], operation="calculate the correlation")
        _check_row_counts_are_equal([self, other])
        _check_column_has_no_missing_values(self, other_columns=[other], operation="calculate the correlation")

        return pl.DataFrame({"a": self._series, "b": other._series}).corr().item(row=1, column="a")

    def distinct_value_count(
        self,
        *,
        ignore_missing_values: bool = True,
    ) -> int:
        """
        Return the number of distinct values in the column.

        Parameters
        ----------
        ignore_missing_values:
            Whether to ignore missing values when counting distinct values.

        Returns
        -------
        distinct_value_count:
            The number of distinct values in the column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, 3, 2, None])
        >>> column.distinct_value_count()
        3

        >>> column.distinct_value_count(ignore_missing_values=False)
        4
        """
        if ignore_missing_values:
            return self._series.drop_nulls().n_unique()
        else:
            return self._series.n_unique()

    def idness(self) -> float:
        """
        Return the idness of this column.

        We define the idness as the number of distinct values (including missing values) divided by the number of rows.
        If the column is empty, the idness is 1.0.

        A high idness indicates that most values in the column are unique. In this case, you must be careful when using
        the column for analysis, as a model might learn a mapping from this column to the target, which might not
        generalize well. You can generally ignore this metric for floating point columns.

        Returns
        -------
        idness:
            The idness of the column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("a", [1, 2, 3])
        >>> column1.idness()
        1.0

        >>> column2 = Column("a", [1, 2, 3, 2])
        >>> column2.idness()
        0.75
        """
        if self.row_count == 0:
            return 1.0  # All values are unique (since there are none)

        return self._series.n_unique() / self.row_count

    def max(self) -> T_co | None:
        """
        Return the maximum value in the column.

        Returns
        -------
        max:
            The maximum value in the column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, 3])
        >>> column.max()
        3
        """
        from polars.exceptions import InvalidOperationError

        try:
            # Fails for Null columns
            return self._series.max()
        except InvalidOperationError:
            return None  # Return None to indicate that we don't know the maximum (consistent with mean and median)

    def mean(self) -> T_co:
        """
        Return the mean of the values in the column.

        The mean is the sum of the values divided by the number of values.

        Returns
        -------
        mean:
            The mean of the values in the column.

        Raises
        ------
        ColumnTypeError
            If the column is not numeric.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, 3])
        >>> column.mean()
        2.0
        """
        _check_column_is_numeric(self, operation="calculate the mean")

        return self._series.mean()

    def median(self) -> T_co:
        """
        Return the median of the values in the column.

        The median is the value in the middle of the sorted list of values. If the number of values is even, the median
        is the mean of the two middle values.

        Returns
        -------
        median:
            The median of the values in the column.

        Raises
        ------
        ColumnTypeError
            If the column is not numeric.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, 3])
        >>> column.median()
        2.0

        >>> column = Column("a", [1, 2, 3, 4])
        >>> column.median()
        2.5
        """
        _check_column_is_numeric(self, operation="calculate the median")

        return self._series.median()

    def min(self) -> T_co | None:
        """
        Return the minimum value in the column.

        Returns
        -------
        min:
            The minimum value in the column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, 3])
        >>> column.min()
        1
        """
        from polars.exceptions import InvalidOperationError

        try:
            # Fails for Null columns
            return self._series.min()
        except InvalidOperationError:
            return None  # Return None to indicate that we don't know the maximum (consistent with mean and median)

    def missing_value_count(self) -> int:
        """
        Return the number of missing values in the column.

        Returns
        -------
        missing_value_count:
            The number of missing values in the column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("a", [1, 2, 3])
        >>> column1.missing_value_count()
        0

        >>> column2 = Column("a", [1, None, 3])
        >>> column2.missing_value_count()
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

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("a", [1, 2, 3])
        >>> column1.missing_value_ratio()
        0.0

        >>> column2 = Column("a", [1, None])
        >>> column2.missing_value_ratio()
        0.5

        >>> column3 = Column("a", [])
        >>> column3.missing_value_ratio()
        1.0
        """
        if self.row_count == 0:
            return 1.0  # All values are missing (since there are none)

        return self._series.null_count() / self.row_count

    @overload
    def mode(
        self,
        *,
        ignore_missing_values: Literal[True] = ...,
    ) -> Sequence[T_co]: ...

    @overload
    def mode(
        self,
        *,
        ignore_missing_values: bool,
    ) -> Sequence[T_co | None]: ...

    def mode(
        self,
        *,
        ignore_missing_values: bool = True,
    ) -> Sequence[T_co | None]:
        """
        Return the mode of the values in the column.

        The mode is the value that appears most frequently in the column. If multiple values occur equally often, all
        of them are returned. The values are sorted in ascending order.

        Parameters
        ----------
        ignore_missing_values:
            Whether to ignore missing values.

        Returns
        -------
        mode:
            The mode of the values in the column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [3, 1, 2, 1, 3])
        >>> column.mode()
        [1, 3]
        """
        import polars as pl

        if self.row_count == 0:
            return []  # polars raises otherwise
        elif self._series.dtype == pl.Null:
            # polars raises otherwise
            if ignore_missing_values:
                return []
            else:
                return [None]

        if ignore_missing_values:
            series = self._series.drop_nulls()
        else:
            series = self._series

        return series.mode().sort().to_list()

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
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("a", [1, 1, 2, 3, None])
        >>> column1.stability()
        0.5

        >>> column2 = Column("a", [1, 1, 1, 1])
        >>> column2.stability()
        1.0

        >>> column3 = Column("a", [])
        >>> column3.stability()
        1.0
        """
        non_missing = self._series.drop_nulls()
        if non_missing.len() == 0:
            return 1.0  # All non-null values are the same (since there is are none)

        # `unique_counts` crashes in polars for boolean columns (https://github.com/pola-rs/polars/issues/16356)
        mode_count = non_missing.value_counts().get_column("count").max()

        return mode_count / non_missing.len()

    def standard_deviation(self) -> float:
        """
        Return the standard deviation of the values in the column.

        The standard deviation is the square root of the variance.

        Returns
        -------
        standard_deviation:
            The standard deviation of the values in the column.

        Raises
        ------
        ColumnTypeError
            If the column is not numeric.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, 3])
        >>> column.standard_deviation()
        1.0
        """
        _check_column_is_numeric(self, operation="calculate the standard deviation")

        return self._series.std()

    def variance(self) -> float:
        """
        Return the variance of the values in the column.

        The variance is the sum of the squared differences from the mean divided by the number of values minus one.

        Returns
        -------
        variance:
            The variance of the values in the column.

        Raises
        ------
        ColumnTypeError
            If the column is not numeric.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, 3])
        >>> column.variance()
        1.0
        """
        _check_column_is_numeric(self, operation="calculate the variance")

        return self._series.var()

    # ------------------------------------------------------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------------------------------------------------------

    def to_list(self) -> list[T_co]:
        """
        Return the values of the column in a list.

        Returns
        -------
        values:
            The values of the column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, 3])
        >>> column.to_list()
        [1, 2, 3]
        """
        return self._series.to_list()

    def to_table(self) -> Table:
        """
        Create a table that contains only this column.

        Returns
        -------
        table:
            The table with this column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, 3])
        >>> column.to_table()
        +-----+
        |   a |
        | --- |
        | i64 |
        +=====+
        |   1 |
        |   2 |
        |   3 |
        +-----+
        """
        from ._table import Table

        return Table._from_polars_lazy_frame(self._lazy_frame)

    # ------------------------------------------------------------------------------------------------------------------
    # IPython integration
    # ------------------------------------------------------------------------------------------------------------------

    def _repr_html_(self) -> str:
        """
        Return a compact HTML representation of the column for IPython.

        Returns
        -------
        html:
            The generated HTML.
        """
        return self._series._repr_html_()
