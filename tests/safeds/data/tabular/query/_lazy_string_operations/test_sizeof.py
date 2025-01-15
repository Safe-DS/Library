import sys

import polars as pl
import pytest

from safeds.data.tabular.containers import Cell
from safeds.data.tabular.containers._lazy_cell import _LazyCell
from safeds.data.tabular.query import StringOperations


@pytest.mark.parametrize(
    "ops",
    [
        Cell.constant("a").str,
        _LazyCell(pl.col("a")).str,
    ],
    ids=[
        "constant",
        "column",
    ],
)
def test_should_be_larger_than_normal_object(ops: StringOperations) -> None:
    assert sys.getsizeof(ops) > sys.getsizeof(object())
