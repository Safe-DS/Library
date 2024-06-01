from typing import Any

import polars as pl
import pytest
from safeds.data.tabular.containers import Table, TemporalCell
from safeds.data.tabular.containers._lazy_temporal_cell import _LazyTemporalCell


@pytest.mark.parametrize(
    ("cell1", "cell2", "expected"),
    [
        (_LazyTemporalCell(pl.col("a")), _LazyTemporalCell(pl.col("a")), True),
        (_LazyTemporalCell(pl.col("a")), _LazyTemporalCell(pl.col("b")), False),
    ],
    ids=[
        "equal",
        "different",
    ],
)
def test_should_return_whether_two_cells_are_equal(cell1: TemporalCell, cell2: TemporalCell, expected: bool) -> None:
    assert (cell1._equals(cell2)) == expected


def test_should_return_true_if_objects_are_identical() -> None:
    cell = _LazyTemporalCell(pl.col("a"))
    assert (cell._equals(cell)) is True


@pytest.mark.parametrize(
    ("cell", "other"),
    [
        (_LazyTemporalCell(pl.col("a")), None),
        (_LazyTemporalCell(pl.col("a")), Table()),
    ],
    ids=[
        "Cell vs. None",
        "Cell vs. Table",
    ],
)
def test_should_return_not_implemented_if_other_is_not_cell(cell: TemporalCell, other: Any) -> None:
    assert (cell._equals(other)) is NotImplemented
