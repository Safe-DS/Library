from typing import Any

import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Row, Table


@pytest.mark.parametrize(
    ("table1", "table2", "expected"),
    [
        (
            TabularDataset({"a": [], "b": []}, "b"),
            TabularDataset({"a": [], "b": []}, "b"),
            True,
        ),
        (
            TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b"),
            TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b"),
            True,
        ),
        (
            TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", ["c"]),
            TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "c", ["b"]),
            False,
        ),
        (
            TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", ["c"]),
            TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6], "d": [7, 8, 9]}, "b", ["d"]),
            False,
        ),
        (
            TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b"),
            TabularDataset({"a": [1, 1, 3], "b": [4, 5, 6]}, "b"),
            False,
        ),
        (
            TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b"),
            TabularDataset({"a": ["1", "2", "3"], "b": [4, 5, 6]}, "b"),
            False,
        ),
        (
            TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", ["c"]),
            TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", ["a"]),
            False,
        ),
    ],
    ids=[
        "rowless table",
        "equal tables",
        "different target",
        "different column names",
        "different values",
        "different types",
        "different features",
    ],
)
def test_should_return_whether_two_tabular_datasets_are_equal(
    table1: TabularDataset,
    table2: TabularDataset,
    expected: bool,
) -> None:
    assert (table1.__eq__(table2)) == expected


@pytest.mark.parametrize(
    ("table", "other"),
    [
        (TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b"), None),
        (TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b"), Row()),
        (TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b"), Table()),
    ],
    ids=[
        "TabularDataset vs. None",
        "TabularDataset vs. Row",
        "TabularDataset vs. Table",
    ],
)
def test_should_return_not_implemented_if_other_is_not_tabular_dataset(table: TabularDataset, other: Any) -> None:
    assert (table.__eq__(other)) is NotImplemented
