from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, overload

from safeds._utils import _structural_hash
from safeds.data.tabular.typing._experimental_polars_data_type import _PolarsDataType
from safeds.exceptions import IndexOutOfBoundsError

from ._column import Column
from ._experimental_vectorized_cell import _VectorizedCell

if TYPE_CHECKING:
    from polars import Series

    from safeds.data.image.containers import Image
    from safeds.data.tabular.typing._experimental_data_type import ExperimentalDataType

    from ._experimental_cell import ExperimentalCell
    from ._experimental_table import ExperimentalTable


T = TypeVar("T")
R = TypeVar("R")


class ExperimentalColumn(Sequence[T]):
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
    >>> from safeds.data.tabular.containers import ExperimentalColumn
    >>> ExperimentalColumn("test", [1, 2, 3])
    Column("test", [1, 2, 3])
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
    def type(self) -> ExperimentalDataType:
        """The type of the column."""
        return _PolarsDataType(self._series.dtype)

    # ------------------------------------------------------------------------------------------------------------------
    # Value operations
    # ------------------------------------------------------------------------------------------------------------------

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

    def remove_duplicate_values(self) -> ExperimentalColumn[T]:
        """
        Return a new column with duplicate values removed.

        The original column is not modified.

        Returns
        -------
        unique_values:
            A new column with duplicate values removed.

        Examples
        --------
        >>> from safeds.data.tabular.containers import ExperimentalColumn
        >>> column = ExperimentalColumn("test", [1, 2, 3, 2, 4, 3])
        >>> column.remove_duplicate_values()
        [1, 2, 3, 4]
        """
        return self._from_polars_series(self._series.unique(maintain_order=True))

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
        [1, 2, 3]
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
        >>> column.transform(lambda cell: cell * 2)
        [2, 4, 6]
        """
        result = transformer(_VectorizedCell(self))
        if not isinstance(result, _VectorizedCell):
            raise TypeError("The transformer must return a cell.")

        return self._from_polars_series(result._series)

    # ------------------------------------------------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------------------------------------------------

    def summarize_statistics(self) -> ExperimentalTable:
        raise NotImplementedError

    def correlation_with(self, other: ExperimentalColumn) -> float:
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

    def mode(self) -> ExperimentalColumn[T]:
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
        x
        """
        from ._experimental_table import ExperimentalTable

        return ExperimentalTable._from_polars_dataframe(self._series.to_frame())

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
        >>> column = ExperimentalColumn("a": [1, 2, 3])
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
