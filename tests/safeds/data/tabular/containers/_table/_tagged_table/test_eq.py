from typing import Any

import pytest
from safeds.data.tabular.containers import Row, Table, TaggedTable


@pytest.mark.parametrize(
    ("table1", "table2", "expected"),
    [
        (TaggedTable({"a": [], "b": []}, "b", ["a"]), TaggedTable({"a": [], "b": []}, "b", ["a"]), True),
        (TaggedTable({"a": [1, 2, 3], "b": [4, 5, 6]}, "b", ["a"]), TaggedTable({"a": [1, 2, 3], "b": [4, 5, 6]}, "b", ["a"]), True),
        (TaggedTable({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", ["a"]), TaggedTable({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "c", ["a"]), False),
        (TaggedTable({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", ["a"]), TaggedTable({"a": [1, 2, 3], "b": [4, 5, 6], "d": [7, 8, 9]}, "b", ["a"]), False),
        (TaggedTable({"a": [1, 2, 3], "b": [4, 5, 6]}, "b", ["a"]), TaggedTable({"a": [1, 1, 3], "b": [4, 5, 6]}, "b", ["a"]), False),
        (TaggedTable({"a": [1, 2, 3], "b": [4, 5, 6]}, "b", ["a"]), TaggedTable({"a": ["1", "2", "3"], "b": [4, 5, 6]}, "b", ["a"]), False),
        (TaggedTable({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", ["a"]), TaggedTable({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", ["c"]), False),
    ],
    ids=[
        "rowless table",
        "equal tables",
        "different target",
        "different column names",
        "different values",
        "different types",
        "different features"
    ],
)
def test_should_return_whether_two_tagged_tables_are_equal(table1: TaggedTable, table2: TaggedTable, expected: bool) -> None:
    assert (table1.__eq__(table2)) == expected


@pytest.mark.parametrize(
    ("table", "other"),
    [
        (TaggedTable({"a": [1, 2, 3], "b": [4, 5, 6]}, "b", ["a"]), None),
        (TaggedTable({"a": [1, 2, 3], "b": [4, 5, 6]}, "b", ["a"]), Row()),
        (TaggedTable({"a": [1, 2, 3], "b": [4, 5, 6]}, "b", ["a"]), Table())
    ],
    ids=[
        "TaggedTable vs. None",
        "TaggedTable vs. Row",
        "TaggedTable vs. Table",
    ],
)
def test_should_return_not_implemented_if_other_is_not_tagged_table(table: TaggedTable, other: Any) -> None:
    assert (table.__eq__(other)) is NotImplemented
