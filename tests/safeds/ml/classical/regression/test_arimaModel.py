import pytest
from safeds.ml.classical.regression import ArimaModel
from safeds.data.tabular.containers import TimeSeries, Table
from syrupy import SnapshotAssertion
import numpy as np
import os

from tests.helpers import resolve_resource_path


def test_arima_model(snapshot_png: SnapshotAssertion) -> None:
    # Create a DataFrame
    _inflation_path = "_datas/US_Inflation_rates.csv"
    np.random.seed(42)
    time_series = TimeSeries.timeseries_from_csv_file(path=resolve_resource_path(_inflation_path), target_name="value",
                                                      time_name="date")
    train_ts, test_ts = time_series.split_rows(0.8)
    model = ArimaModel()
    trained_model = model.fit(train_ts)
    predictions = trained_model.predict(test_ts)
    print(os.getcwd)
    print(predictions)
    assert snapshot_png == trained_model.plot_predictions(test_ts)


def create_test_data() -> TimeSeries:
    return TimeSeries({"time": [1, 2, 3, 4, 5, 6, 7, 8, 9], "value": [1, 2, 3, 4, 5, 6, 7, 8, 9]},
                      time_name="time", target_name="value")


def test_should_succeed_on_valid_data() -> None:
    valid_data = create_test_data()
    model = ArimaModel()
    model.fit(valid_data)
    assert True


def test_should_succeed_on_valid_data_plot() -> None:
    valid_data = create_test_data()
    model = ArimaModel()
    fitted_model = model.fit(valid_data)
    fitted_model.plot_predictions(valid_data)
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
