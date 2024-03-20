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
    time_series = TimeSeries.timeseries_from_csv_file(path=resolve_resource_path(_inflation_path), target_name= "value", time_name="date")
    train_ts, test_ts = time_series.split_rows(0.8)
    model = ArimaModel()
    trained_model = model.fit(train_ts)
    predictions = trained_model.predict(test_ts)
    print(os.getcwd)
    print(predictions)
    assert snapshot_png == trained_model.plot_predictions(test_ts)

