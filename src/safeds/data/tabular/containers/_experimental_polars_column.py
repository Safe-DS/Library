from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, overload

if TYPE_CHECKING:
    from polars import Series

T = TypeVar("T")


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
        return self._series.__len__()

    def __repr__(self) -> str:
        return self._series.__repr__()

    def __sizeof__(self) -> int:
        raise NotImplementedError

    def __str__(self) -> str:
        return self._series.__str__()
