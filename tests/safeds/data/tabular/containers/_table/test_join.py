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
def test_join(
    table_left: Table,
    table_right: Table,
    left_names: list[str],
    right_names: list[str],
    mode: Literal["inner", "left", "outer"],
    table_expected: Table,
) -> None:
    assert table_left.join(table_right, left_names, right_names, mode=mode) == table_expected


def test_join_mismatched_columns() -> None:
    table_left = Table({"a": [1, 2], "b": [3, 4]})
    table_right = Table({"d": [1, 5], "e": [5, 6]})
    left_names = ["a"]
    right_names = ["d", "e"]
    with pytest.raises(ValueError, match="The number of columns to join on must be the same in both tables."):
        table_left.join(table_right, left_names, right_names)


def test_join_check_columns_exist() -> None:
    table_left = Table({"a": [1, 2], "b": [3, 4]})
    table_right = Table({"d": [1, 5], "e": [5, 6]})
    left_names = ["a", "c"]
    right_names = ["d", "e"]
    mode: Literal["inner", "left", "outer"] = "inner"
    with pytest.raises(ColumnNotFoundError):
        table_left.join(table_right, left_names=left_names, right_names=right_names, mode=mode)
    left_names = ["a"]
    right_names = ["d", "f"]
    with pytest.raises(ColumnNotFoundError):
        table_left.join(table_right, left_names=left_names, right_names=right_names, mode=mode)

def test_check_columns_exist() -> None:
    table = Table({"a": [1, 2], "b": [3, 4]})
    _check_columns_exist(table, ["a"])  # Should not raise
    _check_columns_exist(table, ["a", "b"])  # Should not raise
    with pytest.raises(ColumnNotFoundError):
        _check_columns_exist(table, ["c"])  # Should raise

def test_check_columns_exist_in_join() -> None:
    table_left = Table({"a": [1, 2], "b": [3, 4]})
    table_right = Table({"d": [1, 5], "e": [5, 6]})
    left_names = ["a"]
    right_names = ["d"]
    mode: Literal["inner", "left", "outer"] = "inner"
    table_left.join(table_right, left_names=left_names, right_names=right_names, mode=mode)  # Should not raise

def test_invalid_column_name_in_join() -> None:
    table_left = Table({"a": [1, 2], "b": [3, 4]})
    table_right = Table({"d": [1, 5], "e": [5, 6]})
    left_names = ["z"]  # Invalid column
    right_names = ["d"]
    mode: Literal["inner", "left", "outer"] = "inner"
    with pytest.raises(ColumnNotFoundError):
        table_left.join(table_right, left_names=left_names, right_names=right_names, mode=mode)

def test_check_columns_exist_edge_cases() -> None:
    table = Table({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ColumnNotFoundError):
        _check_columns_exist(table, [])  # Empty column list
    with pytest.raises(ColumnNotFoundError):
        _check_columns_exist(table, ["a", "b", "c"])  # Mixed valid and invalid columns
