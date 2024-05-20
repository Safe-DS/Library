import sys
from typing import TYPE_CHECKING, Any

import polars as pl
from safeds.data.tabular.containers._lazy_cell import _LazyCell

if TYPE_CHECKING:
    from safeds.data.tabular.containers import Cell


def test_should_return_size_greater_than_normal_object() -> None:
    cell: Cell[Any] = _LazyCell(pl.col("a"))
    assert sys.getsizeof(cell) > sys.getsizeof(object())
