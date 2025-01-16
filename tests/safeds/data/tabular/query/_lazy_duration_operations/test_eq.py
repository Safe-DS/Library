from typing import Any

import polars as pl
import pytest

from safeds.data.tabular.containers import Cell, Column
from safeds.data.tabular.containers._lazy_cell import _LazyCell
from safeds.data.tabular.query import DurationOperations


@pytest.mark.parametrize(
    ("ops_1", "ops_2", "expected"),
    [
        # equal (duration)
        (
            Cell.duration(hours=1).dur,
            Cell.duration(hours=1).dur,
            True,
        ),
        # equal (column)
        (
            _LazyCell(pl.col("a")).dur,
            _LazyCell(pl.col("a")).dur,
            True,
        ),
        # not equal (different durations)
        (
            Cell.duration(hours=1).dur,
            Cell.duration(hours=2).dur,
            False,
        ),
        # not equal (different columns)
        (
            _LazyCell(pl.col("a")).dur,
            _LazyCell(pl.col("b")).dur,
            False,
        ),
        # not equal (different cell kinds)
        (
            Cell.duration(hours=1).dur,
            _LazyCell(pl.col("a")).dur,
            False,
        ),
    ],
    ids=[
        # Equal
        "equal (duration)",
        "equal (column)",
        # Not equal
        "not equal (different durations)",
        "not equal (different columns)",
        "not equal (different cell kinds)",
    ],
)
def test_should_return_whether_objects_are_equal(
    ops_1: DurationOperations,
    ops_2: DurationOperations,
    expected: bool,
) -> None:
    assert (ops_1.__eq__(ops_2)) == expected


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
def test_should_return_true_if_objects_are_identical(ops: DurationOperations) -> None:
    assert (ops.__eq__(ops)) is True


@pytest.mark.parametrize(
    ("ops", "other"),
    [
        (Cell.duration(hours=1).dur, None),
        (Cell.duration(hours=1).dur, Column("col1", [1])),
    ],
    ids=[
        "DurationOperations vs. None",
        "DurationOperations vs. Column",
    ],
)
def test_should_return_not_implemented_if_other_has_different_type(ops: DurationOperations, other: Any) -> None:
    assert (ops.__eq__(other)) is NotImplemented
