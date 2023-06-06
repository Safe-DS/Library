from collections.abc import Callable

import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "selector", "expected"),
    [
        (
            Table({"col1": [1, 1, 2, 2, 3], "col2": ["a", "b", "c", "d", "e"]}),
            lambda row: row["col1"],
            {
                1: Table({"col1": [1, 1], "col2": ["a", "b"]}),
                2: Table({"col1": [2, 2], "col2": ["c", "d"]}),
                3: Table({"col1": [3], "col2": ["e"]}),
            },
        ),
        (
            Table({"col1": [1, 1, 2, 2, 3], "col2": ["a", "b", "c", "d", 2]}),
            lambda row: row["col1"],
            {
                1: Table({"col1": [1, 1], "col2": ["a", "b"]}),
                2: Table({"col1": [2, 2], "col2": ["c", "d"]}),
                3: Table({"col1": [3], "col2": [2]}),
            },
        ),
        (Table(), lambda row: row["col1"], {}),
        (Table({"col1": [], "col2": []}), lambda row: row["col1"], {}),
    ],
    ids=["select by row1", "different types in column", "empty table", "table with no rows"],
)
def test_group_by(table: Table, selector: Callable, expected: dict) -> None:
    out = table.group_by(selector)
    assert out == expected
