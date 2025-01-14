from typing import Any

import pytest

from safeds.data.tabular.containers import Cell, Column


@pytest.mark.parametrize(
    ("column_1", "column_2", "expected"),
    [
        # equal (no rows)
        (
            Column("col1", []),
            Column("col1", []),
            True,
        ),
        # equal (with data)
        (
            Column("col1", [1]),
            Column("col1", [1]),
            True,
        ),
        # not equal (different column names)
        (
            Column("col1", [1]),
            Column("col2", [1]),
            False,
        ),
        # not equal (different types)
        (
            Column("col1", [1]),
            Column("col1", ["1"]),
            False,
        ),
        # not equal (too few rows)
        (
            Column("col1", [1, 2]),
            Column("col1", [1]),  # Needs at least one value, so the types match
            False,
        ),
        # not equal (too many rows)
        (
            Column("col1", [1]),  # Needs at least one value, so the types match
            Column("col1", [1, 2]),
            False,
        ),
        # not equal (different row order)
        (
            Column("col1", [1, 2]),
            Column("col1", [2, 1]),
            False,
        ),
        # not equal (different values)
        (
            Column("col1", [1, 2]),
            Column("col1", [1, 3]),
            False,
        ),
    ],
    ids=[
        # Equal
        "equal (no rows)",
        "equal (with data)",
        # Not equal
        "not equal (different names)",
        "not equal (different types)",
        "not equal (too few rows)",
        "not equal (too many rows)",
        "not equal (different row order)",
        "not equal (different values)",
    ],
)
def test_should_return_whether_objects_are_equal(column_1: Column, column_2: Column, expected: bool) -> None:
    assert (column_1.__eq__(column_2)) == expected


@pytest.mark.parametrize(
    "column",
    [
        Column("col1", []),
        Column("col1", [1]),
    ],
    ids=[
        "no rows",
        "non-empty",
    ],
)
def test_should_return_true_if_objects_are_identical(column: Column) -> None:
    assert (column.__eq__(column)) is True


@pytest.mark.parametrize(
    ("column", "other"),
    [
        (Column("col1", []), None),
        (Column("col1", []), Cell.from_literal(1)),
    ],
    ids=[
        "Column vs. None",
        "Column vs. Cell",
    ],
)
def test_should_return_not_implemented_if_other_has_different_type(column: Column, other: Any) -> None:
    assert (column.__eq__(other)) is NotImplemented
