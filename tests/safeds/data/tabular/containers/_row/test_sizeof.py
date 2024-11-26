import sys

import polars as pl

from safeds.data.tabular.containers._lazy_vectorized_row import _LazyVectorizedRow


def test_should_return_size_greater_than_normal_object() -> None:
    cell = _LazyVectorizedRow(pl.col("a"))
    assert sys.getsizeof(cell) > sys.getsizeof(object())
