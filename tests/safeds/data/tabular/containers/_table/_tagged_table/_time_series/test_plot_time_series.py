import pytest
from safeds.data.tabular.containers import TimeSeries
from safeds.exceptions import NonNumericColumnError, UnknownColumnNameError
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
    plot = table.plot_time_series_lineplot()
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
        table.plot_time_series_lineplot(x_column_name="feature_1")


def test_should_return_table_both(snapshot_png: SnapshotAssertion) -> None:
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
    plot = table.plot_time_series_lineplot(x_column_name="feature_1", y_column_name="target")
    assert plot == snapshot_png


def test_should_plot_feature_y(snapshot_png: SnapshotAssertion) -> None:
    table = TimeSeries(
        {
            "time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature_1": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "target": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        },
        target_name="target",
        time_name="time",
        feature_names=None,
    )
    plot = table.plot_time_series_lineplot(y_column_name="feature_1")
    assert plot == snapshot_png


def test_should_plot_feature_x(snapshot_png: SnapshotAssertion) -> None:
    table = TimeSeries(
        {
            "time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature_1": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "target": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        },
        target_name="target",
        time_name="time",
        feature_names=None,
    )
    plot = table.plot_time_series_lineplot(x_column_name="feature_1")
    assert plot == snapshot_png


def test_should_plot_feature(snapshot_png: SnapshotAssertion) -> None:
    table = TimeSeries(
        {
            "time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature_1": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            "target": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        },
        target_name="target",
        time_name="time",
        feature_names=None,
    )
    plot = table.plot_time_series_lineplot(x_column_name="feature_1")
    assert plot == snapshot_png


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
            r"Tried to do a numerical operation on one or multiple non-numerical columns: \nThe time series plotted"
            r" column"
            r" contains"
            r" non-numerical columns."
        ),
    ):
        table.plot_time_series_lineplot()


@pytest.mark.parametrize(
    ("time_series", "name", "error", "error_msg"),
    [
        (
            TimeSeries(
                {
                    "time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "feature_1": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                    "target": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                },
                target_name="target",
                time_name="time",
                feature_names=None,
            ),
            "feature_1",
            NonNumericColumnError,
            r"Tried to do a numerical operation on one or multiple non-numerical columns: \nThe time series plotted"
            r" column"
            r" contains"
            r" non-numerical columns.",
        ),
        (
            TimeSeries(
                {
                    "time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "feature_1": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                    "target": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                },
                target_name="target",
                time_name="time",
                feature_names=None,
            ),
            "feature_3",
            UnknownColumnNameError,
            r"Could not find column\(s\) 'feature_3'.",
        ),
    ],
    ids=["feature_not_numerical", "feature_does_not_exist"],
)
def test_should_raise_error_optional_parameter(
    time_series: TimeSeries,
    name: str,
    error: type[Exception],
    error_msg: str,
) -> None:
    with pytest.raises(
        error,
        match=error_msg,
    ):
        time_series.plot_time_series_lineplot(x_column_name=name)


@pytest.mark.parametrize(
    ("time_series", "name", "error", "error_msg"),
    [
        (
            TimeSeries(
                {
                    "time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "feature_1": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                    "target": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                },
                target_name="target",
                time_name="time",
                feature_names=None,
            ),
            "feature_1",
            NonNumericColumnError,
            r"Tried to do a numerical operation on one or multiple non-numerical columns: \nThe time series plotted"
            r" column"
            r" contains"
            r" non-numerical columns.",
        ),
        (
            TimeSeries(
                {
                    "time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "feature_1": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                    "target": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                },
                target_name="target",
                time_name="time",
                feature_names=None,
            ),
            "feature_3",
            UnknownColumnNameError,
            r"Could not find column\(s\) 'feature_3'.",
        ),
    ],
    ids=["feature_not_numerical", "feature_does_not_exist"],
)
def test_should_raise_error_optional_parameter_y(
    time_series: TimeSeries,
    name: str,
    error: type[Exception],
    error_msg: str,
) -> None:
    with pytest.raises(
        error,
        match=error_msg,
    ):
        time_series.plot_time_series_lineplot(y_column_name=name)
