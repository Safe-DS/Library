from typing import Any

import pytest
from safeds.data.tabular.containers import Table, TimeSeries
from safeds.exceptions import (
    DatasetMissesDataError,
    MissingValuesColumnError,
    ModelNotFittedError,
    NonNumericColumnError,
    NonTimeSeriesError,
)
from safeds.ml.classical.regression import ArimaModel

from tests.helpers import resolve_resource_path


def test_arima_model() -> None:
    # Create a DataFrame
    _inflation_path = "_datas/US_Inflation_rates.csv"
    time_series = TimeSeries.timeseries_from_csv_file(
        path=resolve_resource_path(_inflation_path), target_name="value", time_name="date",
    )
    train_ts, test_ts = time_series.split_rows(0.8)
    model = ArimaModel()
    trained_model = model.fit(train_ts)
    trained_model.plot_predictions(test_ts)
    # suggest it ran through
    assert True


def create_test_data() -> TimeSeries:
    return TimeSeries(
        {"time": [1, 2, 3, 4, 5, 6, 7, 8, 9], "value": [1, 2, 3, 4, 5, 6, 7, 8, 9]},
        time_name="time",
        target_name="value",
    )


def test_should_succeed_on_valid_data() -> None:
    valid_data = create_test_data()
    model = ArimaModel()
    model.fit(valid_data)
    assert True


def test_should_not_change_input_regressor() -> None:
    valid_data = create_test_data()
    model = ArimaModel()
    model.fit(valid_data)
    assert not model.is_fitted()


def test_should_not_change_input_table() -> None:
    valid_data = create_test_data()
    valid_data_copy = create_test_data()
    model = ArimaModel()
    model.fit(valid_data)
    assert valid_data_copy == valid_data


def test_should_succeed_on_valid_data_plot() -> None:
    valid_data = create_test_data()
    model = ArimaModel()
    fitted_model = model.fit(valid_data)
    fitted_model.plot_predictions(valid_data)
    assert True


@pytest.mark.parametrize(
    ("invalid_data", "expected_error", "expected_error_msg"),
    [
        (
            Table(
                {
                    "id": [1, 4],
                    "feat1": [1, 5],
                    "feat2": [3, 6],
                    "target": ["0", 1],
                },
            ).time_columns(target_name="target", feature_names=["feat1", "feat2"], time_name="id"),
            NonNumericColumnError,
            r"Tried to do a numerical operation on one or multiple non-numerical columns: \ntarget",
        ),
        (
            Table(
                {
                    "id": [1, 4],
                    "feat1": [1, 5],
                    "feat2": [3, 6],
                    "target": [None, 1],
                },
            ).time_columns(target_name="target", feature_names=["feat1", "feat2"], time_name="id"),
            MissingValuesColumnError,
            r"Tried to do an operation on one or multiple columns containing missing values: \ntarget\nYou can use the Imputer to replace the missing values based on different strategies.\nIf you want toremove the missing values entirely you can use the method `TimeSeries.remove_rows_with_missing_values`.",
        ),
        (
            Table(
                {
                    "id": [],
                    "feat1": [],
                    "feat2": [],
                    "target": [],
                },
            ).time_columns(target_name="target", feature_names=["feat1", "feat2"], time_name="id"),
            DatasetMissesDataError,
            r"Dataset contains no rows",
        ),
    ],
    ids=["non-numerical data", "missing values in data", "no rows in data"],
)
def test_should_raise_on_invalid_data(
    invalid_data: TimeSeries,
    expected_error: Any,
    expected_error_msg: str,
) -> None:
    model = ArimaModel()
    with pytest.raises(expected_error, match=expected_error_msg):
        model.fit(invalid_data)


@pytest.mark.parametrize(
    "table",
    [
        Table(
            {
                "a": [1.0, 0.0, 0.0, 0.0],
                "b": [0.0, 1.0, 1.0, 0.0],
                "c": [0.0, 0.0, 0.0, 1.0],
            },
        ),
    ],
    ids=["untagged_table"],
)
def test_should_raise_if_table_is_not_tagged(table: Table) -> None:
    model = ArimaModel()
    with pytest.raises(NonTimeSeriesError):
        model.fit(table)  # type: ignore[arg-type]


def test_correct_structure_of_time_series() -> None:
    data = create_test_data()
    model = ArimaModel()
    model = model.fit(data)
    predics_ts = model.predict(5)
    assert len(predics_ts.time) == 5
    assert len(predics_ts.target) == 5
    assert predics_ts.time.name == "time"
    assert predics_ts.target.name == "target"


def test_should_raise_if_not_fitted() -> None:
    model = ArimaModel()
    with pytest.raises(ModelNotFittedError):
        model.predict(forecast_horizon=5)


def test_if_fitted_not_fitted() -> None:
    model = ArimaModel()
    assert not model.is_fitted()


def test_if_fitted_fitted() -> None:
    model = ArimaModel()
    model = model.fit(create_test_data())
    assert model.is_fitted()
