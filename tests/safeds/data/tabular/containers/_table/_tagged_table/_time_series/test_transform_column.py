import pytest
from safeds.data.tabular.containers import TimeSeries
from safeds.exceptions import UnknownColumnNameError

from tests.helpers import assert_that_time_series_are_equal

# here is the time column transformable
@pytest.mark.parametrize(
    ("table", "column_name", "table_transformed"),
    [
        (
            TimeSeries({"time":[0, 1, 2],"feature_a": [1, 2, 3], "feature_b": [4, 5, 6], "target": [1, 2, 3]}, "target", "time"),
            "feature_a",
            TimeSeries({"time":[0, 1, 2], "feature_a": [2, 4, 6], "feature_b": [4, 5, 6], "target": [1, 2, 3]}, "target", "time"),
        ),
        (
            TimeSeries({"time":[0, 1, 2], "feature_a": [1, 2, 3], "feature_b": [4, 5, 6], "target": [1, 2, 3]}, "target", "time"),
            "target",
            TimeSeries({"time":[0, 1, 2], "feature_a": [1, 2, 3], "feature_b": [4, 5, 6], "target": [2, 4, 6]}, "target", "time"),
        ),
        (
            TimeSeries(
                {"time":[0, 1, 2], "feature_a": [1, 2, 3], "b": [4, 5, 6], "target": [1, 2, 3]},
                target_name="target",
                time_name= "time",
                feature_names=["feature_a"],
            ),
            "b",
            TimeSeries(
                {"time":[0, 1, 2], "feature_a": [1, 2, 3], "b": [8, 10, 12], "target": [1, 2, 3]},
                target_name="target",
                time_name= "time",
                feature_names=["feature_a"],
            ),
        ),
        (
            TimeSeries(
                {"time": [0, 1, 2], "feature_a": [1, 2, 3], "b": [4, 5, 6], "target": [1, 2, 3]},
                target_name="target",
                time_name="time",
                feature_names=["feature_a"],
            ),
            "time",
            TimeSeries(
                {"time": [0, 2, 4], "feature_a": [1, 2, 3], "b": [4, 5, 6], "target": [1, 2, 3]},
                target_name="target",
                time_name="time",
                feature_names=["feature_a"],
            ),
        ),
    ],
    ids=["transform_feature_column", "transform_target_column", "transform_column_that_is_neither", "transform_time_col"],
)
def test_should_transform_column(table: TimeSeries, column_name: str, table_transformed: TimeSeries) -> None:
    result = table.transform_column(column_name, lambda row: row.get_value(column_name) * 2)
    assert_that_time_series_are_equal(result, table_transformed)


@pytest.mark.parametrize(
    ("table", "column_name"),
    [
        (
            TimeSeries(
                {
                    "time":[0, 1, 2],
                    "A": [1, 2, 3],
                    "B": [4, 5, 6],
                    "C": ["a", "b", "c"],
                },
                "C",
                "time",
            ),
            "D",
        ),
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "A": [1, 2, 3],
                    "B": [4, 5, 6],
                    "C": ["a", "b", "c"],
                },
                target_name="C",
                time_name="time",
                feature_names=["A"],
            ),
            "D",
        ),
    ],
    ids=["has_only_features_and_target", "has_columns_that_are_neither"],
)
def test_should_raise_if_column_not_found(table: TimeSeries, column_name: str) -> None:
    with pytest.raises(UnknownColumnNameError, match=rf"Could not find column\(s\) '{column_name}'"):
        table.transform_column(column_name, lambda row: row.get_value("A") * 2)

