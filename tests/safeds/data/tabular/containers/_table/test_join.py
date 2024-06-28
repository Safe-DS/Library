import pytest  # noqa: I001
from safeds.data.tabular.containers import Table

@pytest.mark.parametrize(
    ("table_left", "table_right", "left_names", "right_names", "mode", "table_expected"), 
    [
        (
            Table({"a": [1, 2], "b": [3, 4]}),
            Table({"d": [1, 5], "e": [5, 6]}),
            ["a"],
            ["d"],
            "outer",
            Table({"a": [1, None, 2], "b": [3, None, 4], "d": [1, 5, None], "e" : [5,6, None]}),
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
            Table({"d": [1, 5], "e": [5, 6],"g": [7, 9]}),
            ["a", "c"],
            ["d", "e"],
            "inner",
            Table({"a": [1], "b": [3], "c": [5], "g":[7]}),
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
def test_join(table_left, table_right, left_names, right_names, mode, table_expected)-> None:
    assert table_left.join(table_right, left_names, right_names, mode=mode) == table_expected