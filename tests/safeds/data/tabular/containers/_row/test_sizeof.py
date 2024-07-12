import sys
from typing import TYPE_CHECKING, Any

import polars as pl
from safeds.data.tabular.containers._lazy_vectorized_row import _LazyVectorizedRow

if TYPE_CHECKING:
    from safeds.data.tabular.containers import Row


def test_should_return_size_greater_than_normal_object() -> None:
    cell: Row[Any] = _LazyVectorizedRow(pl.col("a"))
    assert sys.getsizeof(cell) > sys.getsizeof(object())
