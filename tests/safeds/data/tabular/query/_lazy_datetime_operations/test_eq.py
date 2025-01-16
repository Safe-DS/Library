from typing import Any

import polars as pl
import pytest

from safeds.data.tabular.containers import Cell, Column
from safeds.data.tabular.containers._lazy_cell import _LazyCell
from safeds.data.tabular.query import DatetimeOperations


@pytest.mark.parametrize(
    ("ops_1", "ops_2", "expected"),
    [
        # equal (time)
        (
            Cell.time(1, 0, 0).dt,
            Cell.time(1, 0, 0).dt,
            True,
        ),
        # equal (column)
        (
            _LazyCell(pl.col("a")).dt,
            _LazyCell(pl.col("a")).dt,
            True,
        ),
        # not equal (different times)
        (
            Cell.time(1, 0, 0).dt,
            Cell.time(2, 0, 0).dt,
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
            Cell.time(1, 0, 0).dt,
            _LazyCell(pl.col("a")).dt,
            False,
        ),
    ],
    ids=[
        # Equal
        "equal (time)",
        "equal (column)",
        # Not equal
        "not equal (different times)",
        "not equal (different columns)",
        "not equal (different cell kinds)",
    ],
)
def test_should_return_whether_objects_are_equal(
    ops_1: DatetimeOperations,
    ops_2: DatetimeOperations,
    expected: bool,
) -> None:
    assert (ops_1.__eq__(ops_2)) == expected


@pytest.mark.parametrize(
    "ops",
    [
        Cell.time(1, 0, 0).dt,
        _LazyCell(pl.col("a")).dt,
    ],
    ids=[
        "time",
        "column",
    ],
)
def test_should_return_true_if_objects_are_identical(ops: DatetimeOperations) -> None:
    assert (ops.__eq__(ops)) is True


@pytest.mark.parametrize(
    ("ops", "other"),
    [
        (Cell.time(1, 0, 0).dt, None),
        (Cell.time(1, 0, 0).dt, Column("col1", [1])),
    ],
    ids=[
        "DatetimeOperations vs. None",
        "DatetimeOperations vs. Column",
    ],
)
def test_should_return_not_implemented_if_other_has_different_type(ops: DatetimeOperations, other: Any) -> None:
    assert (ops.__eq__(other)) is NotImplemented
