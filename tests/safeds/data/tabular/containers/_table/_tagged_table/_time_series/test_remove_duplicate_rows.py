import pytest
from safeds.data.tabular.containers import TimeSeries

from tests.helpers import assert_that_time_series_are_equal


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (
            TimeSeries(
                {
                    "time":[0, 0, 1],
                    "feature": [0, 0, 1],
                    "target": [2, 2, 3],
                },
                "target",
                "time",
            ),
            TimeSeries(
                {
                    "time": [0, 1],
                    "feature": [0, 1],
                    "target": [2, 3],
                },
                "target",
                "time",
            ),
        ),
        (
            TimeSeries(
                {
                    "time": [0, 0, 1],
                    "feature": [0, 1, 2],
                    "target": [2, 2, 3],
                },
                "target",
                "time",
            ),
            TimeSeries(
                {
                    "time": [0, 0, 1],
                    "feature": [0, 1, 2],
                    "target": [2, 2, 3],
                },
                "target",
                "time",
            ),
        ),
    ],
    ids=["with_duplicate_rows", "without_duplicate_rows"],
)
def test_should_remove_duplicate_rows(table: TimeSeries, expected: TimeSeries) -> None:
    new_table = table.remove_duplicate_rows()
    assert_that_time_series_are_equal(new_table, expected)
