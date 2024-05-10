from typing import Any

import pytest
from safeds.data.tabular.containers import Column, Table


@pytest.mark.parametrize(
    ("column1", "column2", "expected"),
    [
        (Column("a"), Column("a"), True),
        (Column("a", [1, 2, 3]), Column("a", [1, 2, 3]), True),
        (Column("a"), Column("b"), False),
        (Column("a", [1, 2, 3]), Column("a", [1, 2, 4]), False),
        (Column("a", [1, 2, 3]), Column("a", ["1", "2", "3"]), False),
    ],
    ids=[
        "empty columns",
        "equal columns",
        "different names",
        "different values",
        "different types",
    ],
)
def test_should_return_whether_two_columns_are_equal(column1: Column, column2: Column, expected: bool) -> None:
    assert (column1.__eq__(column2)) == expected


@pytest.mark.parametrize(
    "column",
    [
        Column("a"),
        Column("a", [1, 2, 3]),
    ],
    ids=[
        "empty",
        "non-empty",
    ],
)
def test_should_return_true_if_objects_are_identical(column: Column) -> None:
    assert (column.__eq__(column)) is True


@pytest.mark.parametrize(
    ("column", "other"),
    [
        (Column("a"), None),
        (Column("a", [1, 2, 3]), Table()),
    ],
    ids=[
        "Column vs. None",
        "Column vs. Table",
    ],
)
def test_should_return_not_implemented_if_other_is_not_column(column: Column, other: Any) -> None:
    assert (column.__eq__(other)) is NotImplemented
