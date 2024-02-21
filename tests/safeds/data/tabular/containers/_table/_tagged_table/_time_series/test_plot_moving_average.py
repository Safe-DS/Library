import pytest
from safeds.data.tabular.containers import TimeSeries
from safeds.exceptions import NonNumericColumnError
from syrupy import SnapshotAssertion


def test_should_return_table(snapshot_png: SnapshotAssertion) -> None:
    table = TimeSeries(
        {
            "time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "target": [1, 2, 3, 4, 3, 2, 1, 2, 3, 4],
        },
        target_name="target",
        time_name="time",
        feature_names=None,
    )
    moving_average_plot = table.plot_moving_average(window_size=2)
    assert moving_average_plot == snapshot_png


def test_should_raise_if_column_contains_non_numerical_values() -> None:
    table = TimeSeries(
        {
            "time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "target": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        },
        target_name="target",
        time_name="time",
        feature_names=None,
    )
    with pytest.raises(
        NonNumericColumnError,
        match=(
            r"Tried to do a numerical operation on one or multiple non-numerical columns: \nThis time series target"
            r" contains"
            r" non-numerical columns."
        ),
    ):
        table.plot_moving_average(2)
