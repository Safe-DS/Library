import pytest
from safeds.data.labeled.containers import TabularDataset


@pytest.mark.parametrize(
    ("table1", "table2"),
    [
        (
            TabularDataset({"a": [], "b": []}, "b"),
            TabularDataset({"a": [], "b": []}, "b"),
        ),
        (
            TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b"),
            TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b"),
        ),
        (
            TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b"),
            TabularDataset({"a": [1, 1, 3], "b": [4, 5, 6]}, "b"),
        ),
    ],
    ids=[
        "rowless table",
        "equal tables",
        "different values",
    ],
)
def test_should_return_same_hash_for_equal_tabular_datasets(table1: TabularDataset, table2: TabularDataset) -> None:
    assert hash(table1) == hash(table2)


@pytest.mark.parametrize(
    ("table1", "table2"),
    [
        (
            TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", ["c"]),
            TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "c", ["b"]),
        ),
        (
            TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", ["c"]),
            TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6], "d": [7, 8, 9]}, "b", ["d"]),
        ),
        (
            TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b",),
            TabularDataset({"a": ["1", "2", "3"], "b": [4, 5, 6]}, "b"),
        ),
        (
            TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", ["c"]),
            TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", ["a"]),
        ),
    ],
    ids=[
        "different target",
        "different column names",
        "different types",
        "different features",
    ],
)
def test_should_return_different_hash_for_unequal_tabular_datasets(
    table1: TabularDataset,
    table2: TabularDataset,
) -> None:
    assert hash(table1) != hash(table2)
