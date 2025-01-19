import sys

import polars as pl
import pytest

from safeds.data.tabular.containers import Cell
from safeds.data.tabular.containers._lazy_cell import _LazyCell
from safeds.data.tabular.query import DurationOperations


@pytest.mark.parametrize(
    "ops",
    [
        Cell.duration(hours=1).dur,
        _LazyCell(pl.col("a")).dur,
    ],
    ids=[
        "duration",
        "column",
    ],
)
def test_should_be_larger_than_normal_object(ops: DurationOperations) -> None:
    assert sys.getsizeof(ops) > sys.getsizeof(object())
