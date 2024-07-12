from typing import Any

import polars as pl
import pytest
from safeds.data.tabular.containers import Cell
from safeds.data.tabular.containers._lazy_cell import _LazyCell


def test_should_be_deterministic() -> None:
    cell: Cell[Any] = _LazyCell(pl.col("a"))
    assert hash(cell) == 977452292332124345


@pytest.mark.parametrize(
    ("cell1", "cell2", "expected"),
    [
        (_LazyCell(pl.col("a")), _LazyCell(pl.col("a")), True),
        (_LazyCell(pl.col("a")), _LazyCell(pl.col("b")), False),
    ],
    ids=[
        "equal",
        "different",
    ],
)
def test_should_be_good_hash(cell1: Cell, cell2: Cell, expected: bool) -> None:
    assert (hash(cell1) == hash(cell2)) == expected
