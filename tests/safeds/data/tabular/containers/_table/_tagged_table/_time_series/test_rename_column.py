import pytest
from safeds.data.tabular.containers import TimeSeries

from tests.helpers import assert_that_time_series_are_equal


@pytest.mark.parametrize(
    ("original_table", "old_column_name", "new_column_name", "result_table"),
    [
        (
            TimeSeries(
                {
                    "time" : [0, 1, 2],
                    "feature_old": [0, 1, 2],
                    "no_feature": [2, 3, 4],
                    "target": [3, 4, 5],
                },
                target_name="target",
                time_name= "time",
                feature_names=["feature_old"],
            ),
            "feature_old",
            "feature_new",
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_new": [0, 1, 2],
                    "no_feature": [2, 3, 4],
                    "target": [3, 4, 5],
                },
                target_name="target",
                time_name="time",
                feature_names=["feature_new"],
            ),
        ),
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature": [0, 1, 2],
                    "no_feature": [2, 3, 4],
                    "target_old": [3, 4, 5],
                },
                target_name="target_old",
                time_name="time",
                feature_names=["feature"],
            ),
            "target_old",
            "target_new",
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature": [0, 1, 2],
                    "no_feature": [2, 3, 4],
                    "target_new": [3, 4, 5],
                },
                target_name="target_new",
                time_name = "time",
                feature_names=["feature"],
            ),
        ),
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature": [0, 1, 2],
                    "no_feature_old": [2, 3, 4],
                    "target": [3, 4, 5],
                },
                target_name="target",
                time_name= "time",
                feature_names=["feature"],
            ),
            "no_feature_old",
            "no_feature_new",
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature": [0, 1, 2],
                    "no_feature_new": [2, 3, 4],
                    "target": [3, 4, 5],
                },
                target_name="target",
                time_name="time",
                feature_names=["feature"],
            ),
        ),
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature": [0, 1, 2],
                    "no_feature_old": [2, 3, 4],
                    "target": [3, 4, 5],
                },
                target_name="target",
                time_name= "time",
                feature_names=["feature"],
            ),
            "time",
            "new_time",
            TimeSeries(
                {
                    "new_time": [0, 1, 2],
                    "feature": [0, 1, 2],
                    "no_feature_old": [2, 3, 4],
                    "target": [3, 4, 5],
                },
                target_name="target",
                time_name="new_time",
                feature_names=["feature"],
            ),
        ),
    ],
    ids=["rename_feature_column", "rename_target_column", "rename_non_feature_column", "rename_time_column"],
)
def test_should_rename_column(
    original_table: TimeSeries,
    old_column_name: str,
    new_column_name: str,
    result_table: TimeSeries,
) -> None:
    new_table = original_table.rename_column(old_column_name, new_column_name)
    assert_that_time_series_are_equal(new_table, result_table)
