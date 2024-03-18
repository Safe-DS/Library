import pytest
from safeds.data.tabular.containers import Column, TimeSeries


@pytest.mark.parametrize(
    ("time_series", "time"),
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
                time_name="time",
                feature_names=["A", "B", "C"],
            ),
            Column("time", [0, 1]),
        ),
        (
            TimeSeries(
                {
                    "time": [1, 2],
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
                target_name="T",
                time_name="time",
                feature_names=["A", "C"],
            ),
            Column("time", [1, 2]),
        ),
    ],
    ids=["only_target_and_features", "target_features_and_other"],
)
def test_should_return_features(time_series: TimeSeries, time: Column) -> None:
    assert time_series.time == time
