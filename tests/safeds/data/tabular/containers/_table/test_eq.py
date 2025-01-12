from typing import Any

import pytest

from safeds.data.tabular.containers import Column, Table


@pytest.mark.parametrize(
    ("table_1", "table_2", "expected"),
    [
        # equal (empty)
        (
            Table({}),
            Table({}),
            True,
        ),
        # equal (no rows)
        (
            Table({"col1": []}),
            Table({"col1": []}),
            True,
        ),
        # equal (with data)
        (
            Table({"col1": [1], "col2": [2]}),
            Table({"col1": [1], "col2": [2]}),
            True,
        ),
        # not equal (too few columns)
        (
            Table({"col1": [1]}),
            Table({}),
            False,
        ),
        # not equal (too many columns)
        (
            Table({}),
            Table({"col1": [1]}),
            False,
        ),
        # not equal (different column order)
        (
            Table({"col1": [1], "col2": [2]}),
            Table({"col2": [2], "col1": [1]}),
            False,
        ),
        # not equal (different column names)
        (
            Table({"col1": [1]}),
            Table({"col2": [1]}),
            False,
        ),
        # not equal (different types)
        (
            Table({"col1": [1]}),
            Table({"col1": ["1"]}),
            False,
        ),
        # not equal (too few rows)
        (
            Table({"col1": [1, 2]}),
            Table({"col1": [1]}),  # Needs at least one value, so the types match
            False,
        ),
        # not equal (too many rows)
        (
            Table({"col1": [1]}),  # Needs at least one value, so the types match
            Table({"col1": [1, 2]}),
            False,
        ),
        # not equal (different row order)
        (
            Table({"col1": [1, 2]}),
            Table({"col1": [2, 1]}),
            False,
        ),
        # not equal (different values)
        (
            Table({"col1": [1, 2]}),
            Table({"col1": [1, 3]}),
            False,
        ),
    ],
    ids=[
        # Equal
        "equal (empty)",
        "equal (no rows)",
        "equal (with data)",
        # Not equal because of columns
        "not equal (too few columns)",
        "not equal (too many columns)",
        "not equal (different column order)",
        "not equal (different column names)",
        "not equal (different types)",
        # Not equal because of rows
        "not equal (too few rows)",
        "not equal (too many rows)",
        "not equal (different row order)",
        "not equal (different values)",
    ],
)
def test_should_return_whether_tables_are_equal(table_1: Table, table_2: Table, expected: bool) -> None:
    assert (table_1.__eq__(table_2)) == expected


@pytest.mark.parametrize(
    "table",
    [
        Table({}),
        Table({"col1": []}),
        Table({"col1": [1]}),
    ],
    ids=[
        "empty",
        "no rows",
        "non-empty",
    ],
)
def test_should_return_true_if_tables_are_identical(table: Table) -> None:
    assert (table.__eq__(table)) is True


@pytest.mark.parametrize(
    ("table", "other"),
    [
        (Table({}), None),
        (Table({}), Column("col1", [])),
    ],
    ids=[
        "Table vs. None",
        "Table vs. Column",
    ],
)
def test_should_return_not_implemented_if_other_is_not_table(table: Table, other: Any) -> None:
    assert (table.__eq__(other)) is NotImplemented
