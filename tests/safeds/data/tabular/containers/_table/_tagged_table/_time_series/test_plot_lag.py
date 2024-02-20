import pytest
from safeds.data.tabular.containers import Table, TimeSeries
from syrupy import SnapshotAssertion
from safeds.exceptions import IllegalSchemaModificationError, NonNumericColumnError

from tests.helpers import assert_that_time_series_are_equal


def test_should_return_table(snapshot_png: SnapshotAssertion) -> None:
    table = TimeSeries(
        {
            "time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "target": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        },
        target_name="target",
        time_name="time",
        feature_names=None, )
    lag_plot = table.plot_lagplot(lag=1)
    assert lag_plot == snapshot_png

def test_should_raise_if_column_contains_non_numerical_values() -> None:
    table = TimeSeries(
        {
            "time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "target": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        },
        target_name="target",
        time_name="time",
        feature_names=None, )
    with pytest.raises(
        NonNumericColumnError,
        match=(
            r"Tried to do a numerical operation on one or multiple non-numerical columns: \nThis time series target contains"
            r" non-numerical columns."
        ),
    ):
        table.plot_lagplot(2)

