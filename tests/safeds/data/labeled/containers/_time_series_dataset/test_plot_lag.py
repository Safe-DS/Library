import pytest
from safeds.data.labeled.containers import TimeSeriesDataset
from safeds.exceptions import NonNumericColumnError
from syrupy import SnapshotAssertion


def test_should_return_table(snapshot_png_image: SnapshotAssertion) -> None:
    table = TimeSeriesDataset(
        {
            "time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "target": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        },
        target_name="target",
        time_name="time",
    )
    lag_plot = table.plot_lagplot(lag=1)
    assert lag_plot == snapshot_png_image


def test_should_raise_if_column_contains_non_numerical_values() -> None:
    table = TimeSeriesDataset(
        {
            "time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "target": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        },
        target_name="target",
        time_name="time",
    )
    with pytest.raises(
        NonNumericColumnError,
        match=(
            r"Tried to do a numerical operation on one or multiple non-numerical columns: \nThis time series target"
            r" contains"
            r" non-numerical columns."
        ),
    ):
        table.plot_lagplot(2)
