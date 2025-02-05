from typing import Any

import pytest

from safeds.data.labeled.containers import TimeSeriesDataset
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table1", "table2", "expected"),
    [
        (
            TimeSeriesDataset({"a": [], "b": [], "c": []}, "b", window_size=1),
            TimeSeriesDataset({"a": [], "b": [], "c": []}, "b", window_size=1),
            True,
        ),
        (
            TimeSeriesDataset({"a": [1, 2, 3], "b": [4, 5, 6], "c": [4, 5, 6]}, "b", window_size=1),
            TimeSeriesDataset({"a": [1, 2, 3], "b": [4, 5, 6], "c": [4, 5, 6]}, "b", window_size=1),
            True,
        ),
        (
            TimeSeriesDataset(
                {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]},
                "b",
                window_size=1,
                extra_names=["c"],
            ),
            TimeSeriesDataset(
                {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]},
                "c",
                window_size=1,
                extra_names=["b"],
            ),
            False,
        ),
        (
            TimeSeriesDataset(
                {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]},
                "b",
                window_size=1,
                extra_names=["c"],
            ),
            TimeSeriesDataset(
                {"a": [1, 2, 3], "b": [4, 5, 6], "d": [7, 8, 9]},
                "b",
                window_size=1,
                extra_names=["d"],
            ),
            False,
        ),
        (
            TimeSeriesDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b", window_size=1),
            TimeSeriesDataset({"a": [1, 1, 3], "b": [4, 5, 6]}, "b", window_size=1),
            False,
        ),
        (
            TimeSeriesDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b", window_size=1),
            TimeSeriesDataset({"a": ["1", "2", "3"], "b": [4, 5, 6]}, "b", window_size=1),
            False,
        ),
        (
            TimeSeriesDataset(
                {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]},
                "b",
                window_size=1,
                extra_names=["c"],
            ),
            TimeSeriesDataset(
                {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]},
                "b",
                window_size=1,
                extra_names=["a"],
            ),
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
    table1: TimeSeriesDataset,
    table2: TimeSeriesDataset,
    expected: bool,
) -> None:
    assert (table1.__eq__(table2)) == expected


@pytest.mark.parametrize(
    ("table", "other"),
    [
        (
            TimeSeriesDataset({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0, 0, 0]}, "b", window_size=1),
            None,
        ),
        (
            TimeSeriesDataset({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0, 0, 0]}, "b", window_size=1),
            Table({}),
        ),
    ],
    ids=[
        "TabularDataset vs. None",
        "TabularDataset vs. Table",
    ],
)
def test_should_return_not_implemented_if_other_is_not_tabular_dataset(table: TimeSeriesDataset, other: Any) -> None:
    assert (table.__eq__(other)) is NotImplemented
