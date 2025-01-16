from typing import Any

import polars as pl
import pytest

from safeds.data.tabular.containers import Cell, Column
from safeds.data.tabular.containers._lazy_cell import _LazyCell
from safeds.data.tabular.query import MathOperations


@pytest.mark.parametrize(
    ("ops_1", "ops_2", "expected"),
    [
        # equal (constant)
        (
            Cell.constant(1).math,
            Cell.constant(1).math,
            True,
        ),
        # equal (column)
        (
            _LazyCell(pl.col("a")).math,
            _LazyCell(pl.col("a")).math,
            True,
        ),
        # not equal (different constant)
        (
            Cell.constant(1).math,
            Cell.constant(2).math,
            False,
        ),
        # not equal (different columns)
        (
            _LazyCell(pl.col("a")).math,
            _LazyCell(pl.col("b")).math,
            False,
        ),
        # not equal (different cell kinds)
        (
            Cell.constant(1).math,
            _LazyCell(pl.col("a")).math,
            False,
        ),
    ],
    ids=[
        # Equal
        "equal (constant)",
        "equal (column)",
        # Not equal
        "not equal (different constants)",
        "not equal (different columns)",
        "not equal (different cell kinds)",
    ],
)
def test_should_return_whether_objects_are_equal(
    ops_1: MathOperations,
    ops_2: MathOperations,
    expected: bool,
) -> None:
    assert (ops_1.__eq__(ops_2)) == expected


@pytest.mark.parametrize(
    "ops",
    [
        Cell.constant(1).math,
        _LazyCell(pl.col("a")).math,
    ],
    ids=[
        "duration",
        "column",
    ],
)
def test_should_return_true_if_objects_are_identical(ops: MathOperations) -> None:
    assert (ops.__eq__(ops)) is True


@pytest.mark.parametrize(
    ("ops", "other"),
    [
        (Cell.constant(1).math, None),
        (Cell.constant(1).math, Column("col1", [1])),
    ],
    ids=[
        "MathOperations vs. None",
        "MathOperations vs. Column",
    ],
)
def test_should_return_not_implemented_if_other_has_different_type(ops: MathOperations, other: Any) -> None:
    assert (ops.__eq__(other)) is NotImplemented
