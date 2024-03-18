import pytest
from safeds.data.tabular.containers import TimeSeries

from tests.helpers import assert_that_time_series_are_equal


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (
            TimeSeries(
                {
                    "time": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    "feature": [1.0, 11.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    "target": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                },
                "target",
                "time",
            ),
            TimeSeries(
                {
                    "time": [0, 2, 3, 4, 5, 6, 7, 8, 9],
                    "feature": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    "target": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                },
                "target",
                "time",
            ),
        ),
        (
            TimeSeries(
                {
                    "time": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    "feature": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    "target": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                },
                "target",
                "time",
            ),
            TimeSeries(
                {
                    "time": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    "feature": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    "target": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                },
                "target",
                "time",
            ),
        ),
    ],
    ids=["with_outliers", "no_outliers"],
)
def test_should_remove_rows_with_outliers(table: TimeSeries, expected: TimeSeries) -> None:
    new_table = table.remove_rows_with_outliers()
    assert_that_time_series_are_equal(new_table, expected)
