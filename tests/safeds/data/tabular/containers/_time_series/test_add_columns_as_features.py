import pytest
from safeds.data.tabular.containers import Column, Table, TimeSeries
from safeds.exceptions import ColumnSizeError, DuplicateColumnNameError

from tests.helpers import assert_that_time_series_are_equal


@pytest.mark.parametrize(
    ("time_series", "columns", "time_series_with_new_columns"),
    [
        (
            Table({"time": [0, 1], "f1": [1, 2], "target": [2, 3]})._time_columns(
                target_name="target",
                time_name="time",
                feature_names=["f1"],
            ),
            [Column("f2", [4, 5]), Column("f3", [6, 7])],
            Table({"time": [0, 1], "f1": [1, 2], "target": [2, 3], "f2": [4, 5], "f3": [6, 7]})._time_columns(
                target_name="target",
                time_name="time",
                feature_names=["f1", "f2", "f3"],
            ),
        ),
        (
            Table({"time": [0, 1], "f1": [1, 2], "target": [2, 3]})._time_columns(
                target_name="target",
                time_name="time",
                feature_names=["f1"],
            ),
            Table.from_columns([Column("f2", [4, 5]), Column("f3", [6, 7])]),
            Table({"time": [0, 1], "f1": [1, 2], "target": [2, 3], "f2": [4, 5], "f3": [6, 7]})._time_columns(
                target_name="target",
                time_name="time",
                feature_names=["f1", "f2", "f3"],
            ),
        ),
        (
            Table({"time": [0, 1], "f1": [1, 2], "target": [2, 3], "other": [0, -1]})._time_columns(
                target_name="target",
                time_name="time",
                feature_names=["f1"],
            ),
            Table.from_columns([Column("f2", [4, 5]), Column("f3", [6, 7])]),
            Table(
                {
                    "time": [0, 1],
                    "f1": [1, 2],
                    "target": [2, 3],
                    "other": [0, -1],
                    "f2": [4, 5],
                    "f3": [6, 7],
                },
            )._time_columns(
                target_name="target",
                time_name="time",
                feature_names=["f1", "f2", "f3"],
            ),
        ),
    ],
    ids=["new columns as feature", "table added as features", "table contains a non feature/target column"],
)
def test_add_columns_as_features(
    time_series: TimeSeries,
    columns: list[Column] | Table,
    time_series_with_new_columns: TimeSeries,
) -> None:
    assert_that_time_series_are_equal(time_series.add_columns_as_features(columns), time_series_with_new_columns)


@pytest.mark.parametrize(
    ("time_series", "columns", "error_msg"),
    [
        (
            TimeSeries(
                {"time": [0, 1, 2], "A": [1, 2, 3], "B": [4, 5, 6]},
                target_name="B",
                time_name="time",
                feature_names=["A"],
            ),
            [Column("A", [7, 8, 9]), Column("D", [10, 11, 12])],
            r"Column 'A' already exists.",
        ),
    ],
    ids=["column_already_exist"],
)
def test_add_columns_raise_duplicate_column_name_if_column_already_exist(
    time_series: TimeSeries,
    columns: list[Column] | Table,
    error_msg: str,
) -> None:
    with pytest.raises(DuplicateColumnNameError, match=error_msg):
        time_series.add_columns_as_features(columns)


@pytest.mark.parametrize(
    ("time_series", "columns", "error_msg"),
    [
        (
            TimeSeries(
                {"time": [0, 1, 2], "A": [1, 2, 3], "B": [4, 5, 6]},
                target_name="B",
                time_name="time",
                feature_names=["A"],
            ),
            [Column("C", [5, 7, 8, 9]), Column("D", [4, 10, 11, 12])],
            r"Expected a column of size 3 but got column of size 4.",
        ),
    ],
    ids=["columns_are_oversize"],
)
def test_should_raise_column_size_error_if_columns_are_oversize(
    time_series: TimeSeries,
    columns: list[Column] | Table,
    error_msg: str,
) -> None:
    with pytest.raises(ColumnSizeError, match=error_msg):
        time_series.add_columns_as_features(columns)
