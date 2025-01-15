from typing import Any

import polars as pl
import pytest

from safeds.data.tabular.containers import Cell, Column
from safeds.data.tabular.containers._lazy_cell import _LazyCell
from safeds.data.tabular.query import StringOperations


@pytest.mark.parametrize(
    ("ops_1", "ops_2", "expected"),
    [
        # equal (constant)
        (
            Cell.constant("a").str,
            Cell.constant("a").str,
            True,
        ),
        # equal (column)
        (
            _LazyCell(pl.col("a")).str,
            _LazyCell(pl.col("a")).str,
            True,
        ),
        # not equal (different constant values)
        (
            Cell.constant("a").str,
            Cell.constant("b").str,
            False,
        ),
        # not equal (different columns)
        (
            _LazyCell(pl.col("a")).str,
            _LazyCell(pl.col("b")).str,
            False,
        ),
        # not equal (different cell kinds)
        (
            Cell.constant("a").str,
            _LazyCell(pl.col("a")).str,
            False,
        ),
    ],
    ids=[
        # Equal
        "equal (constant)",
        "equal (column)",
        # Not equal
        "not equal (different constant values)",
        "not equal (different columns)",
        "not equal (different cell kinds)",
    ],
)
def test_should_return_whether_objects_are_equal(
    ops_1: StringOperations,
    ops_2: StringOperations,
    expected: bool,
) -> None:
    assert (ops_1.__eq__(ops_2)) == expected


@pytest.mark.parametrize(
    "ops",
    [
        Cell.constant("a").str,
        _LazyCell(pl.col("a")).str,
    ],
    ids=[
        "constant",
        "column",
    ],
)
def test_should_return_true_if_objects_are_identical(ops: StringOperations) -> None:
    assert (ops.__eq__(ops)) is True


@pytest.mark.parametrize(
    ("ops", "other"),
    [
        (Cell.constant("a").str, None),
        (Cell.constant("a").str, Column("col1", [1])),
    ],
    ids=[
        "StringOperations vs. None",
        "StringOperations vs. Column",
    ],
)
def test_should_return_not_implemented_if_other_has_different_type(ops: StringOperations, other: Any) -> None:
    assert (ops.__eq__(other)) is NotImplemented
