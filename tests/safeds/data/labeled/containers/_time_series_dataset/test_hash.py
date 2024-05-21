import pytest
from safeds.data.labeled.containers import TimeSeriesDataset


@pytest.mark.parametrize(
    ("table1", "table2"),
    [
        (
            TimeSeriesDataset({"a": [], "b": []}, "b", "a", window_size=1),
            TimeSeriesDataset({"a": [], "b": []}, "b", "a", window_size=1),
        ),
        (
            TimeSeriesDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b", "a", window_size=1),
            TimeSeriesDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b", "a", window_size=1),
        ),
        (
            TimeSeriesDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b", "a", window_size=1),
            TimeSeriesDataset({"a": [1, 1, 3], "b": [4, 5, 6]}, "b", "a", window_size=1),
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
            TimeSeriesDataset(
                {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]},
                "b",
                "a",
                window_size=1,
                extra_names=["c"],
            ),
            TimeSeriesDataset(
                {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]},
                "c",
                "a",
                window_size=1,
                extra_names=["b"],
            ),
        ),
        (
            TimeSeriesDataset(
                {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]},
                "b",
                "a",
                window_size=1,
                extra_names=["c"],
            ),
            TimeSeriesDataset(
                {"a": [1, 2, 3], "b": [4, 5, 6], "d": [7, 8, 9]},
                "b",
                "a",
                window_size=1,
                extra_names=["d"],
            ),
        ),
        (
            TimeSeriesDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b", "a", window_size=1),
            TimeSeriesDataset({"a": ["1", "2", "3"], "b": [4, 5, 6]}, "b", "a", window_size=1),
        ),
        (
            TimeSeriesDataset(
                {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]},
                "b",
                "a",
                window_size=1,
                extra_names=["c"],
            ),
            TimeSeriesDataset(
                {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]},
                "b",
                "c",
                window_size=1,
                extra_names=["a"],
            ),
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
