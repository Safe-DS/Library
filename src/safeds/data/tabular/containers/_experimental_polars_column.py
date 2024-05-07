from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, overload

from ._column import Column

if TYPE_CHECKING:
    from polars import Series

    from safeds.data.image.containers import Image
    from safeds.data.tabular.typing import ColumnType

    from ._experimental_polars_table import ExperimentalPolarsTable

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
        The data. If None, an empty column is created.

    Examples
    --------
    >>> from safeds.data.tabular.containers import Column
    >>> column = Column("test", [1, 2, 3])
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
        raise NotImplementedError

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> ExperimentalPolarsColumn[T]: ...

    def __getitem__(self, index: int | slice) -> T | ExperimentalPolarsColumn[T]:
        return self._series.__getitem__(index)

    def __hash__(self) -> int:
        raise NotImplementedError

    def __iter__(self) -> Iterator[T]:
        return self._series.__iter__()

    def __len__(self) -> int:
        return self.number_of_rows

    def __repr__(self) -> str:
        return self._series.__repr__()

    def __sizeof__(self) -> int:
        raise NotImplementedError

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

    def get_unique_values(self) -> list[T]:
        raise NotImplementedError

    def get_value(self, index: int) -> T:
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------------------------------------------------------

    def all(self, predicate: Callable[[T], bool]) -> bool:
        raise NotImplementedError

    def any(self, predicate: Callable[[T], bool]) -> bool:
        raise NotImplementedError

    def count(self, predicate: Callable[[T], bool]) -> int:
        raise NotImplementedError

    def none(self, predicate: Callable[[T], bool]) -> bool:
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------------------------------------------------------

    def remove_duplicate_values(self) -> ExperimentalPolarsColumn[T]:
        raise NotImplementedError

    def rename(self, new_name: str) -> ExperimentalPolarsColumn[T]:
        raise NotImplementedError

    def transform(self, transformer: Callable[[T], R]) -> ExperimentalPolarsColumn[R]:
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
        raise NotImplementedError

    def mean(self) -> T:
        raise NotImplementedError

    def median(self) -> T:
        raise NotImplementedError

    def min(self) -> T:
        raise NotImplementedError

    def missing_value_count(self) -> int:
        raise NotImplementedError

    def missing_value_ratio(self) -> float:
        raise NotImplementedError

    def mode(self) -> T:
        raise NotImplementedError

    def stability(self) -> float:
        raise NotImplementedError

    def standard_deviation(self) -> float:
        raise NotImplementedError

    def variance(self) -> float:
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError
