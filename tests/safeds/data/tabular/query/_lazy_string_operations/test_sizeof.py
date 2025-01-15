import sys

import polars as pl

from safeds.data.tabular.query._lazy_string_operations import _LazyStringOperations


def test_should_return_size_greater_than_normal_object() -> None:
    cell = _LazyStringOperations(pl.col("a"))
    assert sys.getsizeof(cell) > sys.getsizeof(object())
