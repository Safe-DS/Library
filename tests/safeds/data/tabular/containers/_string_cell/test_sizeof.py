import sys

import polars as pl
from safeds.data.tabular.containers._lazy_string_cell import _LazyStringCell


def test_should_return_size_greater_than_normal_object() -> None:
    cell = _LazyStringCell(pl.col("a"))
    assert sys.getsizeof(cell) > sys.getsizeof(object())
