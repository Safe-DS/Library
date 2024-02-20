from safeds.data.tabular.containers import TimeSeries
from syrupy import SnapshotAssertion


def test_should_return_table(snapshot_png: SnapshotAssertion) -> None:
    table = TimeSeries(
        {
            "time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "target": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        },
        target_name="target",
        time_name="time",
        feature_names=None,
    )
    lag_plot = table.plot_lag(lag=1)
    assert lag_plot == snapshot_png
