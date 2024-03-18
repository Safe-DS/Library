
import pytest
from safeds.data.tabular.containers import TimeSeries


@pytest.mark.parametrize(
    ("table1", "table2"),
    [
        (
            TimeSeries({"a": [], "b": [], "c": []}, "b", "c", ["a"]),
            TimeSeries({"a": [], "b": [], "c": []}, "b", "c", ["a"]),
        ),
        (
            TimeSeries({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", "c", ["a"]),
            TimeSeries({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", "c", ["a"]),
        ),
        (
            TimeSeries({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", "c", ["a"]),
            TimeSeries({"a": [1, 1, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", "c", ["a"]),
        ),
    ],
    ids=[
        "rowless table",
        "equal tables",
        "different values",
    ],
)
def test_should_return_same_hash_for_equal_time_series(table1: TimeSeries, table2: TimeSeries) -> None:
    assert hash(table1) == hash(table2)


@pytest.mark.parametrize(
    ("table1", "table2"),
    [
        (
            TimeSeries({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [10, 11, 12]}, "b", "d", ["a"]),
            TimeSeries({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [10, 11, 12]}, "c", "d", ["a"]),
        ),
        (
            TimeSeries({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [10, 11, 12]}, "b", "c", ["a"]),
            TimeSeries({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "e": [10, 11, 12]}, "b", "c", ["a"]),
        ),
        (
            TimeSeries({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", "c", ["a"]),
            TimeSeries({"a": ["1", "2", "3"], "b": [4, 5, 6], "c": [7, 8, 9]}, "b", "c", ["a"]),
        ),
        (
            TimeSeries({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [10, 11, 12]}, "b", "d", ["a"]),
            TimeSeries({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [10, 11, 12]}, "b", "d", ["c"]),
        ),
        (
            TimeSeries({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [10, 11, 12]}, "b", "d", ["a"]),
            TimeSeries({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [10, 11, 12]}, "b", "c", ["a"]),
        ),
    ],
    ids=[
        "different target",
        "different column names",
        "different types",
        "different features",
        "different time",
    ],
)
def test_should_return_different_hash_for_unequal_time_series(table1: TimeSeries, table2: TimeSeries) -> None:
    assert hash(table1) != hash(table2)
