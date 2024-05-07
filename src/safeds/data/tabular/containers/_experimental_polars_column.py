from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, overload

from safeds._utils import _structural_hash
from safeds.exceptions import IndexOutOfBoundsError

from ._column import Column
from ._experimental_polars_table import ExperimentalPolarsTable
from ._experimental_vectorized_cell import _VectorizedCell

if TYPE_CHECKING:
    from polars import Series

    from safeds.data.image.containers import Image
    from safeds.data.tabular.typing import ColumnType

    from ._experimental_polars_cell import ExperimentalPolarsCell


T = TypeVar("T")
R = TypeVar("R")


class ExperimentalPolarsColumn(Sequence[T]):
    """
    A column is a named, one-dimensional collection of homogeneous values.

    Parameters
    ----------
    name:
        The name of the column.
    data:
        The data of the column. If None, an empty column is created.

    Examples
    --------
    >>> from safeds.data.tabular.containers import ExperimentalPolarsColumn
    >>> column = ExperimentalPolarsColumn("test", [1, 2, 3])
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _from_polars_series(data: Series) -> ExperimentalPolarsColumn:
        result = object.__new__(ExperimentalPolarsColumn)
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
        if not isinstance(other, ExperimentalPolarsColumn):
            return NotImplemented
        if self is other:
            return True
        return self._series.equals(other._series)

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> ExperimentalPolarsColumn[T]: ...

    def __getitem__(self, index: int | slice) -> T | ExperimentalPolarsColumn[T]:
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
        return self._series.__repr__()

    def __sizeof__(self) -> int:
        return self._series.estimated_size()

    def __str__(self) -> str:
        return self._series.__str__()

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def name(self) -> str:
        """The name of the column."""
        return self._series.name

    @property
    def number_of_rows(self) -> int:
        """The number of rows in the column."""
        return self._series.len()

    @property
    def type(self) -> ColumnType:  # TODO: rethink return type
        """The type of the column."""
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------
    # Value operations
    # ------------------------------------------------------------------------------------------------------------------

    def get_value(self, index: int) -> T:
        """
        Return the column value at specified index.

        Indexing starts at 0.

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
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("test", [1, 2, 3])
        >>> column.get_value(1)
        2
        """
        if index < 0 or index >= self.number_of_rows:
            raise IndexOutOfBoundsError(index)

        return self._series[index]

    # ------------------------------------------------------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------------------------------------------------------

    def all(self, predicate: Callable[[ExperimentalPolarsCell[T]], ExperimentalPolarsCell[bool]]) -> bool:
        import polars as pl

        result = predicate(_VectorizedCell(self))
        if not isinstance(result, _VectorizedCell) or not result._series.dtype == pl.Boolean:
            raise ValueError("The predicate must return a boolean cell.")

        return result._series.all()

    def any(self, predicate: Callable[[ExperimentalPolarsCell[T]], ExperimentalPolarsCell[bool]]) -> bool:
        import polars as pl

        result = predicate(_VectorizedCell(self))
        if not isinstance(result, _VectorizedCell) or not result._series.dtype == pl.Boolean:
            raise ValueError("The predicate must return a boolean cell.")

        return result._series.any()

    def count(self, predicate: Callable[[ExperimentalPolarsCell[T]], ExperimentalPolarsCell[bool]]) -> int:
        import polars as pl

        result = predicate(_VectorizedCell(self))
        if not isinstance(result, _VectorizedCell) or not result._series.dtype == pl.Boolean:
            raise ValueError("The predicate must return a boolean cell.")

        return result._series.sum()

    def none(self, predicate: Callable[[ExperimentalPolarsCell[T]], ExperimentalPolarsCell[bool]]) -> bool:
        import polars as pl

        result = predicate(_VectorizedCell(self))
        if not isinstance(result, _VectorizedCell) or not result._series.dtype == pl.Boolean:
            raise ValueError("The predicate must return a boolean cell.")

        return (~result._series).all()

    # ------------------------------------------------------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------------------------------------------------------

    def remove_duplicate_values(self) -> ExperimentalPolarsColumn[T]:
        """
        Return a list of all unique values in the column.

        Returns
        -------
        unique_values:
            List of unique values in the column.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalPolarsColumn
        >>> column = ExperimentalPolarsColumn("test", [1, 2, 3, 2, 4, 3])
        >>> column.remove_duplicate_values()
        [1, 2, 3, 4]
        """
        return self._from_polars_series(self._series.unique(maintain_order=True))

    def rename(self, new_name: str) -> ExperimentalPolarsColumn[T]:
        return self._from_polars_series(self._series.rename(new_name))

    def transform(
        self,
        transformer: Callable[[ExperimentalPolarsCell[T]], ExperimentalPolarsCell[R]],
    ) -> ExperimentalPolarsColumn[R]:
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------------------------------------------------

    def summarize_statistics(self) -> ExperimentalPolarsTable:
        raise NotImplementedError

    def correlation_with(self, other: ExperimentalPolarsColumn) -> float:
        raise NotImplementedError

    def idness(self) -> float:
        raise NotImplementedError

    def max(self) -> T:
        return self._series.max()

    def mean(self) -> T:
        return self._series.mean()

    def median(self) -> T:
        return self._series.median()

    def min(self) -> T:
        return self._series.min()

    def missing_value_count(self) -> int:
        return self._series.null_count()

    def missing_value_ratio(self) -> float:
        if self.number_of_rows == 0:
            return 0.0

        return self._series.null_count() / self.number_of_rows

    def mode(self) -> ExperimentalPolarsColumn[T]:
        return self._from_polars_series(self._series.mode())

    def stability(self) -> float:
        raise NotImplementedError

    def standard_deviation(self) -> float:
        return self._series.std()

    def variance(self) -> float:
        return self._series.var()

    # ------------------------------------------------------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------------------------------------------------------

    def plot_boxplot(self) -> Image:
        raise NotImplementedError

    def plot_histogram(self) -> Image:
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------------------------------------------------------

    def to_table(self) -> ExperimentalPolarsTable:
        """
        Create a table that contains only this column.

        Returns
        -------
        table:
            The table with this column.
        """
        return ExperimentalPolarsTable._from_polars_dataframe(self._series.to_frame())

    def temporary_to_old_column(self) -> Column:
        """
        Convert the column to the old column format. This method is temporary and will be removed in a later version.

        Returns
        -------
        old_column:
            The column in the old format.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalPolarsColumn
        >>> column = ExperimentalPolarsColumn("a": [1, 2, 3])
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
