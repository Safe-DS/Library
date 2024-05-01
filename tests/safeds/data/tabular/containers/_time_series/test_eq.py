from typing import Any

import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Row, Table, TimeSeries


@pytest.mark.parametrize(
    ("table1", "table2", "expected"),
    [
        (
            TimeSeries({"a": [], "b": [], "c": []}, "b", "c", ["a"]),
            TimeSeries({"a": [], "b": [], "c": []}, "b", "c", ["a"]),
            True,
        ),
        (
            TimeSeries({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", "c", ["a"]),
            TimeSeries({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", "c", ["a"]),
            True,
        ),
        (
            TimeSeries({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [10, 11, 12]}, "b", "d", ["a"]),
            TimeSeries({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [10, 11, 12]}, "c", "d", ["a"]),
            False,
        ),
        (
            TimeSeries({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [10, 11, 12]}, "b", "c", ["a"]),
            TimeSeries({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "e": [10, 11, 12]}, "b", "c", ["a"]),
            False,
        ),
        (
            TimeSeries({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", "c", ["a"]),
            TimeSeries({"a": [1, 1, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", "c", ["a"]),
            False,
        ),
        (
            TimeSeries({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", "c", ["a"]),
            TimeSeries({"a": ["1", "2", "3"], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", "c", ["a"]),
            False,
        ),
        (
            TimeSeries({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [10, 11, 12]}, "b", "d", ["a"]),
            TimeSeries({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [10, 11, 12]}, "b", "d", ["c"]),
            False,
        ),
        (
            TimeSeries({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [10, 11, 12]}, "b", "d", ["a"]),
            TimeSeries({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [10, 11, 12]}, "b", "c", ["a"]),
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
        "different time",
    ],
)
def test_should_return_whether_two_tabular_datasets_are_equal(
    table1: TimeSeries,
    table2: TimeSeries,
    expected: bool,
) -> None:
    assert (table1.__eq__(table2)) == expected


@pytest.mark.parametrize(
    "table1",
    [TimeSeries({"a": [], "b": [], "c": []}, "b", "c", ["a"])],
    ids=[
        "any",
    ],
)
def test_should_return_true_if_objects_are_identical(table1: TimeSeries) -> None:
    assert (table1.__eq__(table1)) is True


@pytest.mark.parametrize(
    ("table", "other"),
    [
        (TimeSeries({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", "c", ["a"]), None),
        (TimeSeries({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", "c", ["a"]), Row()),
        (TimeSeries({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", "c", ["a"]), Table()),
        (
            TimeSeries({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", "c", ["a"]),
            TabularDataset({"a": [1, 2, 3], "b": [4, 5, 6]}, "b"),
        ),
    ],
    ids=[
        "TimeSeries vs. None",
        "TimeSeries vs. Row",
        "TimeSeries vs. Table",
        "TimeSeries vs. TabularDataset",
    ],
)
def test_should_return_not_implemented_if_other_is_not_time_series(table: TimeSeries, other: Any) -> None:
    assert (table.__eq__(other)) is NotImplemented
