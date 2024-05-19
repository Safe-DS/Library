from typing import Any

import polars as pl
import pytest
from safeds.data.tabular.containers import Cell, Table
from safeds.data.tabular.containers._lazy_cell import _LazyCell


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
def test_should_return_whether_two_cells_are_equal(cell1: Cell, cell2: Cell, expected: bool) -> None:
    assert (cell1._equals(cell2)) == expected


def test_should_return_true_if_objects_are_identical() -> None:
    cell = _LazyCell(pl.col("a"))
    assert (cell._equals(cell)) is True


@pytest.mark.parametrize(
    ("cell", "other"),
    [
        (_LazyCell(pl.col("a")), None),
        (_LazyCell(pl.col("a")), Table()),
    ],
    ids=[
        "Cell vs. None",
        "Cell vs. Table",
    ],
)
def test_should_return_not_implemented_if_other_is_not_cell(cell: Cell, other: Any) -> None:
    assert (cell._equals(other)) is NotImplemented
