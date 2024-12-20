from typing import Literal

import pytest

from safeds.data.tabular.containers import Table
from safeds.exceptions import ColumnNotFoundError


@pytest.mark.parametrize(
    ("table_left", "table_right", "left_names", "right_names", "mode", "table_expected"),
    [
        (
            Table({"a": [1, 2], "b": [3, 4]}),
            Table({"d": [1, 5], "e": [5, 6]}),
            ["a"],
            ["d"],
            "outer",
            Table({"a": [1, None, 2], "b": [3, None, 4], "d": [1, 5, None], "e": [5, 6, None]}),
        ),
        (
            Table({"a": [1, 2], "b": [3, 4]}),
            Table({"d": [1, 5], "e": [5, 6]}),
            ["a"],
            ["d"],
            "left",
            Table({"a": [1, 2], "b": [3, 4], "e": [5, None]}),
        ),
        (
            Table({"a": [1, 2], "b": [3, 4]}),
            Table({"d": [1, 5], "e": [5, 6]}),
            ["a"],
            ["d"],
            "right",
            Table({"b": [3, None], "d": [1, 5], "e": [5, 6]}),
        ),
        (
            Table({"a": [1, 2], "b": [3, 4]}),
            Table({"d": [1, 5], "e": [5, 6]}),
            ["a"],
            ["d"],
            "inner",
            Table({"a": [1], "b": [3], "e": [5]}),
        ),
        (
            Table({"a": [1, 2], "b": [3, 4], "c": [5, 6]}),
            Table({"d": [1, 5], "e": [5, 6], "g": [7, 9]}),
            ["a", "c"],
            ["d", "e"],
            "inner",
            Table({"a": [1], "b": [3], "c": [5], "g": [7]}),
        ),
        (
            Table({"a": [1, 2], "b": [3, 4]}),
            Table({"d": [1, 5], "e": [5, 6]}),
            ["b"],
            ["e"],
            "inner",
            Table({"a": [], "b": [], "d": []}),
        ),
    ],
)
def test_should_join_two_tables(
    table_left: Table,
    table_right: Table,
    left_names: list[str],
    right_names: list[str],
    mode: Literal["inner", "left", "right", "outer"],
    table_expected: Table,
) -> None:
    assert table_left.join(table_right, left_names, right_names, mode=mode) == table_expected


def test_should_raise_if_columns_are_mismatched() -> None:
    table_left = Table({"a": [1, 2], "b": [3, 4]})
    table_right = Table({"d": [1, 5], "e": [5, 6]})
    left_names = ["a"]
    right_names = ["d", "e"]
    with pytest.raises(ValueError, match="The number of columns to join on must be the same in both tables."):
        table_left.join(table_right, left_names, right_names)


@pytest.mark.parametrize(
    ("table_left", "table_right", "left_names", "right_names"),
    [
        (Table({"a": [1, 2], "b": [3, 4]}), Table({"d": [1, 5], "e": [5, 6]}), ["c"], ["d"]),
        (Table({"a": [1, 2], "b": [3, 4]}), Table({"d": [1, 5], "e": [5, 6]}), ["a"], ["f"]),
    ],
    ids=[
        "wrong_left_name",
        "wrong_right_name",
    ],
)
def test_should_raise_if_columns_are_missing(
    table_left: Table,
    table_right: Table,
    left_names: list[str],
    right_names: list[str],
) -> None:
    with pytest.raises(ColumnNotFoundError):
        table_left.join(table_right, left_names=left_names, right_names=right_names)
