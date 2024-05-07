import pytest
from safeds.data.labeled.containers import TimeSeriesDataset


@pytest.mark.parametrize(
    ("table1", "table2"),
    [
        (
            TimeSeriesDataset({"a": [], "b": []}, "b", "a"),
            TimeSeriesDataset({"a": [], "b": []}, "b", "a"),
        ),
        (
            TimeSeriesDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b", "a"),
            TimeSeriesDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b", "a"),
        ),
        (
            TimeSeriesDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b", "a"),
            TimeSeriesDataset({"a": [1, 1, 3], "b": [4, 5, 6]}, "b", "a"),
        ),
    ],
    ids=[
        "rowless table",
        "equal tables",
        "different values",
    ],
)
def test_should_return_same_hash_for_equal_tabular_datasets(
    table1: TimeSeriesDataset,
    table2: TimeSeriesDataset,
) -> None:
    assert hash(table1) == hash(table2)


@pytest.mark.parametrize(
    ("table1", "table2"),
    [
        (
            TimeSeriesDataset({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", "a", ["c"]),
            TimeSeriesDataset({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "c", "a", ["b"]),
        ),
        (
            TimeSeriesDataset({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", "a", ["c"]),
            TimeSeriesDataset({"a": [1, 2, 3], "b": [4, 5, 6], "d": [7, 8, 9]}, "b", "a", ["d"]),
        ),
        (
            TimeSeriesDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b", "a"),
            TimeSeriesDataset({"a": ["1", "2", "3"], "b": [4, 5, 6]}, "b", "a"),
        ),
        (
            TimeSeriesDataset({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", "a", ["c"]),
            TimeSeriesDataset({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", "c", ["a"]),
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
    table1: TimeSeriesDataset,
    table2: TimeSeriesDataset,
) -> None:
    assert hash(table1) != hash(table2)
