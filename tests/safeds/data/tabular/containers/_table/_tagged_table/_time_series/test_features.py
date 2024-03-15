import pytest
from safeds.data.tabular.containers import Table, TimeSeries


@pytest.mark.parametrize(
    ("time_series", "features"),
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
                feature_names=["A", "B", "C"]
            ),
            Table({"A": [1, 4], "B": [2, 5], "C": [3, 6]}),
        ),
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
                feature_names=["A", "C"],
            ),
            Table({"A": [1, 4], "C": [3, 6]}),
        ),
    ],
    ids=["only_target_and_features", "target_features_and_other"],
)
def test_should_return_features(time_series: TimeSeries, features: Table) -> None:
    assert time_series._features == features
