from typing import Any

import polars as pl
import pytest

from safeds.data.tabular.containers import Cell, Column
from safeds.data.tabular.containers._lazy_cell import _LazyCell
from safeds.data.tabular.query import TemporalOperations


@pytest.mark.parametrize(
    ("ops_1", "ops_2", "expected"),
    [
        # equal (duration)
        (
            Cell.duration(hours=1).dt,
            Cell.duration(hours=1).dt,
            True,
        ),
        # equal (column)
        (
            _LazyCell(pl.col("a")).dt,
            _LazyCell(pl.col("a")).dt,
            True,
        ),
        # not equal (different durations)
        (
            Cell.duration(hours=1).dt,
            Cell.duration(hours=2).dt,
            False,
        ),
        # not equal (different columns)
        (
            _LazyCell(pl.col("a")).dt,
            _LazyCell(pl.col("b")).dt,
            False,
        ),
        # not equal (different cell kinds)
        (
            Cell.duration(hours=1).dt,
            _LazyCell(pl.col("a")).dt,
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
    ops_1: TemporalOperations,
    ops_2: TemporalOperations,
    expected: bool,
) -> None:
    assert (ops_1.__eq__(ops_2)) == expected


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
def test_should_return_true_if_objects_are_identical(ops: TemporalOperations) -> None:
    assert (ops.__eq__(ops)) is True


@pytest.mark.parametrize(
    ("ops", "other"),
    [
        (Cell.duration(hours=1).dt, None),
        (Cell.duration(hours=1).dt, Column("col1", [1])),
    ],
    ids=[
        "TemporalOperations vs. None",
        "TemporalOperations vs. Column",
    ],
)
def test_should_return_not_implemented_if_other_has_different_type(ops: TemporalOperations, other: Any) -> None:
    assert (ops.__eq__(other)) is NotImplemented
