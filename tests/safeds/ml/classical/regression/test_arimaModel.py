import pytest
from safeds.ml.classical.regression import ArimaModel
from safeds.data.tabular.containers import TimeSeries, Table
from syrupy import SnapshotAssertion
import numpy as np


def test_arima_model(snapshot_png: SnapshotAssertion) -> None:
    # Create a DataFrame
    np.random.seed(42)
    time_series = TimeSeries.timeseries_from_csv_file("C:/Users/ettel/PycharmProjects/Library/tests/resources/_datas/US_Inflation_rates.csv",                                                  target_name= "value", time_name="date")
    train_ts, test_ts = time_series.split_rows(0.8)
    model = ArimaModel()
    trained_model = model.fit(train_ts)
    predictions = trained_model.predict(test_ts)
    print(predictions)
    assert snapshot_png == trained_model.plot_predictions(test_ts)
    assert False

