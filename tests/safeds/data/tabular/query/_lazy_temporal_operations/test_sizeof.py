import sys

import polars as pl
import pytest

from safeds.data.tabular.containers import Cell
from safeds.data.tabular.containers._lazy_cell import _LazyCell
from safeds.data.tabular.query import TemporalOperations


@pytest.mark.parametrize(
    "ops",
    [
        Cell.duration(hours=1).dt,
        _LazyCell(pl.col("a")).dt,
    ],
    ids=[
        "duration",
        "column",
    ],
)
def test_should_be_larger_than_normal_object(ops: TemporalOperations) -> None:
    assert sys.getsizeof(ops) > sys.getsizeof(object())
