from typing import Any

import polars as pl
import pytest

from safeds.data.tabular.containers import Cell, Column
from safeds.data.tabular.containers._lazy_cell import _LazyCell


@pytest.mark.parametrize(
    ("cell_1", "cell_2", "expected"),
    [
        # equal (constant)
        (
            Cell.constant(1),
            Cell.constant(1),
            True,
        ),
        # equal (date, int)
        (
            Cell.date(2025, 1, 15),
            Cell.date(2025, 1, 15),
            True,
        ),
        # equal (date, column)
        (
            Cell.date(_LazyCell(pl.col("a")), 1, 15),
            Cell.date(_LazyCell(pl.col("a")), 1, 15),
            True,
        ),
        # equal (column)
        (
            _LazyCell(pl.col("a")),
            _LazyCell(pl.col("a")),
            True,
        ),
        # not equal (different constant values)
        (
            Cell.constant(1),
            Cell.constant(2),
            False,
        ),
        # not equal (different constant types)
        (
            Cell.constant(1),
            Cell.constant("1"),
            False,
        ),
        # not equal (different dates, int)
        (
            Cell.date(2025, 1, 15),
            Cell.date(2024, 1, 15),
            False,
        ),
        # not equal (different dates, column)
        (
            Cell.date(_LazyCell(pl.col("a")), 1, 15),
            Cell.date(_LazyCell(pl.col("b")), 1, 15),
            False,
        ),
        # not equal (different columns)
        (
            _LazyCell(pl.col("a")),
            _LazyCell(pl.col("b")),
            False,
        ),
        # not equal (different cell kinds)
        (
            Cell.date(23, 1, 15),
            Cell.time(23, 1, 15),
            False,
        ),
    ],
    ids=[
        # Equal
        "equal (constant)",
        "equal (date, int)",
        "equal (date, column)",
        "equal (column)",
        # Not equal
        "not equal (different constant values)",
        "not equal (different constant types)",
        "not equal (different dates, int)",
        "not equal (different dates, column)",
        "not equal (different columns)",
        "not equal (different cell kinds)",
    ],
)
def test_should_return_whether_objects_are_equal(cell_1: Cell, cell_2: Cell, expected: bool) -> None:
    assert (cell_1._equals(cell_2)) == expected


@pytest.mark.parametrize(
    "cell",
    [
        Cell.constant(1),
        Cell.date(2025, 1, 15),
        _LazyCell(pl.col("a")),
    ],
    ids=[
        "constant",
        "date",
        "column",
    ],
)
def test_should_return_true_if_objects_are_identical(cell: Cell) -> None:
    assert (cell._equals(cell)) is True


@pytest.mark.parametrize(
    ("cell", "other"),
    [
        (Cell.constant(1), None),
        (Cell.constant(1), Column("col1", [1])),
    ],
    ids=[
        "Cell vs. None",
        "Cell vs. Column",
    ],
)
def test_should_return_not_implemented_if_other_has_different_type(cell: Cell, other: Any) -> None:
    assert (cell._equals(other)) is NotImplemented
