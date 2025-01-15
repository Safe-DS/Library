import sys

import polars as pl
import pytest

from safeds.data.tabular.containers import Cell
from safeds.data.tabular.containers._lazy_cell import _LazyCell


@pytest.mark.parametrize(
    "cell",
    [
        Cell.constant(1),
        Cell.date(2025, 1, 15),
        Cell.date(_LazyCell(pl.col("a")), 1, 15),
        _LazyCell(pl.col("a")),
    ],
    ids=[
        "constant",
        "date, int",
        "date, column",
        "column",
    ],
)
def test_should_be_larger_than_normal_object(cell: Cell) -> None:
    assert sys.getsizeof(cell) > sys.getsizeof(object())
