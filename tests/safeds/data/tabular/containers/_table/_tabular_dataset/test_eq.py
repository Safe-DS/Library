from typing import Any

import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Row, Table


@pytest.mark.parametrize(
    ("table1", "table2", "expected"),
    [
        (TabularDataset({"a": [], "b": []}, "b", ["a"]), TabularDataset({"a": [], "b": []}, "b", ["a"]), True),
        (TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b", ["a"]), TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b", ["a"]), True),
        (TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", ["a"]), TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "c", ["a"]), False),
        (TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", ["a"]), TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6], "d": [7, 8, 9]}, "b", ["a"]), False),
        (TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b", ["a"]), TabularDataset({"a": [1, 1, 3], "b": [4, 5, 6]}, "b", ["a"]), False),
        (TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b", ["a"]), TabularDataset({"a": ["1", "2", "3"], "b": [4, 5, 6]}, "b", ["a"]), False),
        (TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", ["a"]), TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", ["c"]), False),
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
def test_should_return_whether_two_tabular_datasets_are_equal(table1: TabularDataset, table2: TabularDataset, expected: bool) -> None:
    assert (table1.__eq__(table2)) == expected


@pytest.mark.parametrize(
    ("table", "other"),
    [
        (TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b", ["a"]), None),
        (TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b", ["a"]), Row()),
        (TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b", ["a"]), Table())
    ],
    ids=[
        "TabularDataset vs. None",
        "TabularDataset vs. Row",
        "TabularDataset vs. Table",
    ],
)
def test_should_return_not_implemented_if_other_is_not_tabular_dataset(table: TabularDataset, other: Any) -> None:
    assert (table.__eq__(other)) is NotImplemented
