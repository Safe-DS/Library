import pytest
from safeds.data.tabular.containers import Column, TimeSeries

from tests.helpers import assert_that_time_series_are_equal


@pytest.mark.parametrize(
    ("time_series", "columns", "expected_time_series"),
    [
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_1": [0, 1, 2],
                    "target": [3, 4, 5],
                },
                "target",
                "time",
                None,
            ),
            [Column("other", [6, 7, 8]), Column("other2", [9, 6, 3])],
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_1": [0, 1, 2],
                    "target": [3, 4, 5],
                    "other": [6, 7, 8],
                    "other2": [9, 6, 3],
                },
                "target",
                "time",
                None,
            ),
        ),
    ],
    ids=["add_columns_as_non_feature"],
)
def test_should_add_columns(
    time_series: TimeSeries,
    columns: list[Column],
    expected_time_series: TimeSeries,
) -> None:
    assert_that_time_series_are_equal(time_series.add_columns(columns), expected_time_series)
