import pytest
from safeds.data.tabular.containers import Column, TimeSeries

from tests.helpers import assert_that_time_series_are_equal


@pytest.mark.parametrize(
    ("time_series", "column", "expected_time_series"),
    [
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_1": [0, 1, 2],
                    "target": [3, 4, 5],
                },
                target_name="target",
                time_name="time",
                feature_names=None,
            ),
            Column("other", [6, 7, 8]),
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_1": [0, 1, 2],
                    "target": [3, 4, 5],
                    "other": [6, 7, 8],
                },
                target_name="target",
                time_name="time",
                feature_names=None,
            ),
        ),
    ],
    ids=["add_column_as_non_feature"],
)
def test_should_add_column(time_series: TimeSeries, column: Column, expected_time_series: TimeSeries) -> None:
    assert_that_time_series_are_equal(time_series.add_column(column), expected_time_series)
