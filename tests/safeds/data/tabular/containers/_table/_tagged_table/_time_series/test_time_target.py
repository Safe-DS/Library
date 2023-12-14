import pytest
from safeds.data.tabular.containers import Column, TimeSeries
#test

@pytest.mark.parametrize(
    ("time_series", "target_column", "time_column"),
    [
        (
            TimeSeries(
                {
                    "time": [0, 1],
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
                target_name="T",
                time_name="time"
            ),
            Column("T", [0, 1]),
            Column("time", [0, 1]),
        ),
    ],
    ids=["target"],
)
def test_should_return_target(time_series: TimeSeries, target_column: Column, time_column) -> None:
    assert time_series.target == target_column
    assert time_series.time == time_column
