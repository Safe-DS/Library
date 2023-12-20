import pytest
from safeds.data.tabular.containers import TimeSeries

from tests.helpers import assert_that_time_series_are_equal


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature": [0.0, None, 2.0],
                    "target": [3.0, 4.0, 5.0],
                },
                "target",
                "time",
            ),
            TimeSeries(
                {
                    "time": [0, 2],
                    "feature": [0.0, 2.0],
                    "target": [3.0, 5.0],
                },
                "target",
                "time",
            ),
        ),
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature": [0.0, 1.0, 2.0],
                    "target": [3.0, 4.0, 5.0],
                },
                "target",
                "time",
            ),
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature": [0.0, 1.0, 2.0],
                    "target": [3.0, 4.0, 5.0],
                },
                "target",
                "time",
            ),
        ),
    ],
    ids=["with_missing_values", "without_missing_values"],
)
def test_should_remove_rows_with_missing_values(table: TimeSeries, expected: TimeSeries) -> None:
    new_table = table.remove_rows_with_missing_values()
    assert_that_time_series_are_equal(new_table, expected)
