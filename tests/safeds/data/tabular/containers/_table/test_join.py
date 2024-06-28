import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import ColumnNotFoundError


@pytest.mark.parametrize(
    ("table_left, table_right, left_names, right_names, mode_, table_expected", 
    [
        (
            Table({"a": [1,2], "b": [3,4]}),
            Table({"d": [1,5], "e": [5,6]}),
            ["c"],
            ["d"],
            "outer",
            Table({"a": [1,2,3], "b": [3,4,None], "e": [5, None, None]}),
        ),
            (
            Table({"a": [1,2], "b": [3,4]}),
            Table({"d": [1,5], "e": [5,6]}),
            ["a"],
            ["d"],
            "left",
            Table({"a": [1,2], "b": [5,6], "c": [5, None]}),
        ),
        (
            Table({"a": [1,2], "b": [3,4]}),
            Table({"d": [1,5], "e": [5,6]}),
            ["a"],
            ["d"],
            "inner",
            Table({"a": [1], "b": [3], "e": [5]}),
        ),
        (
            Table({"a": [1,2], "b": [3,4]}),
            Table({"d": [1,5], "e": [5,6]}),
            ["a"],
            ["d"],
            "right",
            Table({"a": [1,3], "b": [3,None], "e": [5,6]}),
        ),
        (
            Table({"a": [1,2], "b": [3,4], "c": [5,6]}),
            Table({"d": [1,5], "e": [5,6],"g": [7,9]}),
            ["a", "c"],
            ["d", "e"],
            "inner",
            Table({"a": [1], "b": [3], "c": [5], "g":[7]}),
        ),
        (
            Table({"a": [1,2], "b": [3,4]}),
            Table({"d": [1,5], "e": [5,6]}),
            ["b"],
            ["e"],
            "inner",
            Table({"a": [], "b": [], "d": []}),
        ),
        (
            Table({"a": [1,2], "b": [3,4]}),
            Table({"d": [], "e": []}),
            ["b"],
            [],
            "inner",
            Table({"a": [], "b": [], "d": []}),
        ),
    ],
    ),
)
def test_join(table_left, table_right, left_names, right_names, mode_, table_expected):
    assert table_left.join(table_right, left_names, right_names, mode=mode_) == table_expected
