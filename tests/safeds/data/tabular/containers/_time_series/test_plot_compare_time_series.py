import pytest
from safeds.data.tabular.containers import TimeSeries
from safeds.exceptions import NonNumericColumnError, UnknownColumnNameError
from syrupy import SnapshotAssertion


def create_time_series_list() -> list[TimeSeries]:
    table1 = TimeSeries(
        {
            "time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "target": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        },
        target_name="target",
        time_name="time",
        feature_names=None,
    )
    table2 = TimeSeries(
        {
            "time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature_1": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            "target": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        },
        target_name="target",
        time_name="time",
        feature_names=None,
    )
    return [table1, table2]


def create_invalid_time_series_list() -> list[TimeSeries]:
    table1 = TimeSeries(
        {
            "time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "target": ["9", 10, 11, 12, 13, 14, 15, 16, 17, 18],
        },
        target_name="target",
        time_name="time",
        feature_names=None,
    )
    table2 = TimeSeries(
        {
            "time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature_1": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            "target": ["4", 5, 6, 7, 8, 9, 10, 11, 12, 13],
        },
        target_name="target",
        time_name="time",
        feature_names=None,
    )
    return [table1, table2]


def test_legit_compare(snapshot_png: SnapshotAssertion) -> None:
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
    plot = table.plot_compare_time_series(create_time_series_list())
    assert plot == snapshot_png



def test_should_raise_if_column_contains_non_numerical_values_x() -> None:
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
            r"Tried to do a numerical operation on one or multiple non-numerical columns: \nThe time series plotted"
            r" column"
            r" contains"
            r" non-numerical columns."
        ),
    ):
        table.plot_compare_time_series(create_time_series_list())


def test_with_non_valid_list() -> None:
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
    with pytest.raises(
        NonNumericColumnError,
        match=(
            r"Tried to do a numerical operation on one or multiple non-numerical columns: \nThe time series plotted"
            r" column"
            r" contains"
            r" non-numerical columns."
        ),
    ):
        table.plot_compare_time_series(create_invalid_time_series_list())
