from typing import Any

import pytest
from safeds.data.tabular.containers import Column, Row


@pytest.mark.parametrize(
    ("row1", "row2", "expected"),
    [
        (Row({}), Row({}), True),
        (Row({"A": 1}), Row({"A": 1}), True),
        (Row({"A": 1}), Row({"A": 2}), False),
        (Row({"A": 1, "B": 2, "C": 3}), Row({"A": 2, "B": 4, "C": 3}), False),
        (Row({"A": 1, "B": 2, "C": 3}), Row({"A": "1", "B": "2", "C": "3"}), False),
    ],
    ids=[
        "empty Row",
        "equal Row",
        "different names",
        "different values",
        "different types",
    ],
)
def test_should_return_whether_two_columns_are_equal(row1: Row, row2: Row, expected: bool) -> None:
    assert (row1.__eq__(row2)) == expected


@pytest.mark.parametrize(
    "row",
    [
        Row({}),
        Row({"A": 1, "B": 2, "C": 3}),
    ],
    ids=[
        "empty",
        "non-empty",
    ],
)
def test_should_return_true_if_objects_are_identical(row: Row) -> None:
    assert (row.__eq__(row)) is True


@pytest.mark.parametrize(
    ("row", "other"),
    [
        (Row({}), None),
        (Row({"A": 1}), Column("col1")),
    ],
    ids=[
        "Row vs. None",
        "Row vs. Column",
    ],
)
def test_should_return_not_implemented_if_other_is_not_column(row: Row, other: Any) -> None:
    assert (row.__eq__(other)) is NotImplemented
