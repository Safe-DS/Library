from typing import Any

import polars as pl
import pytest

from safeds.data.tabular.containers import Table
from safeds.data.tabular.query import TemporalOperations
from safeds.data.tabular.query._lazy_temporal_operations import _LazyTemporalOperations


@pytest.mark.parametrize(
    ("cell1", "cell2", "expected"),
    [
        (_LazyTemporalOperations(pl.col("a")), _LazyTemporalOperations(pl.col("a")), True),
        (_LazyTemporalOperations(pl.col("a")), _LazyTemporalOperations(pl.col("b")), False),
    ],
    ids=[
        "equal",
        "different",
    ],
)
def test_should_return_whether_two_cells_are_equal(
    cell1: TemporalOperations,
    cell2: TemporalOperations,
    expected: bool,
) -> None:
    assert (cell1.__eq__(cell2)) == expected


def test_should_return_true_if_objects_are_identical() -> None:
    cell = _LazyTemporalOperations(pl.col("a"))
    assert (cell.__eq__(cell)) is True


@pytest.mark.parametrize(
    ("cell", "other"),
    [
        (_LazyTemporalOperations(pl.col("a")), None),
        (_LazyTemporalOperations(pl.col("a")), Table({})),
    ],
    ids=[
        "Cell vs. None",
        "Cell vs. Table",
    ],
)
def test_should_return_not_implemented_if_other_is_not_cell(cell: TemporalOperations, other: Any) -> None:
    assert (cell.__eq__(other)) is NotImplemented
