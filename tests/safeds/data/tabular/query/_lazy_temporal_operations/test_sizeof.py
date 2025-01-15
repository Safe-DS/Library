import sys

import polars as pl

from safeds.data.tabular.query._lazy_temporal_operations import _LazyTemporalOperations


def test_should_return_size_greater_than_normal_object() -> None:
    cell = _LazyTemporalOperations(pl.col("a"))
    assert sys.getsizeof(cell) > sys.getsizeof(object())
