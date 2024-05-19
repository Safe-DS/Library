from typing import Any

import polars as pl
import pytest
from safeds.data.tabular.containers import StringCell, Table
from safeds.data.tabular.containers._lazy_string_cell import _LazyStringCell


@pytest.mark.parametrize(
    ("cell1", "cell2", "expected"),
    [
        (_LazyStringCell(pl.col("a")), _LazyStringCell(pl.col("a")), True),
        (_LazyStringCell(pl.col("a")), _LazyStringCell(pl.col("b")), False),
    ],
    ids=[
        "equal",
        "different",
    ],
)
def test_should_return_whether_two_cells_are_equal(cell1: StringCell, cell2: StringCell, expected: bool) -> None:
    assert (cell1._equals(cell2)) == expected


def test_should_return_true_if_objects_are_identical() -> None:
    cell = _LazyStringCell(pl.col("a"))
    assert (cell._equals(cell)) is True


@pytest.mark.parametrize(
    ("cell", "other"),
    [
        (_LazyStringCell(pl.col("a")), None),
        (_LazyStringCell(pl.col("a")), Table()),
    ],
    ids=[
        "Cell vs. None",
        "Cell vs. Table",
    ],
)
def test_should_return_not_implemented_if_other_is_not_cell(cell: StringCell, other: Any) -> None:
    assert (cell._equals(other)) is NotImplemented
