import sys

import polars as pl
import pytest

from safeds.data.tabular.containers import Cell
from safeds.data.tabular.containers._lazy_cell import _LazyCell
from safeds.data.tabular.query import MathOperations


@pytest.mark.parametrize(
    "ops",
    [
        Cell.constant(1).math,
        _LazyCell(pl.col("a")).math,
    ],
    ids=[
        "constant",
        "column",
    ],
)
def test_should_be_larger_than_normal_object(ops: MathOperations) -> None:
    assert sys.getsizeof(ops) > sys.getsizeof(object())
