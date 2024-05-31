import sys

import polars as pl
from safeds.data.tabular.containers._lazy_temporal_cell import _LazyTemporalCell


def test_should_return_size_greater_than_normal_object() -> None:
    cell = _LazyTemporalCell(pl.col("a"))
    assert sys.getsizeof(cell) > sys.getsizeof(object())
