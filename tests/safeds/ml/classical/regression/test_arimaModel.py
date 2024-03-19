import pytest
from safeds.ml.classical.regression import ArimaModel
from safeds.data.tabular.containers import TimeSeries, Table, Column
from syrupy import SnapshotAssertion
import pandas as pd
import numpy as np


def test_arimaModel(snapshot_png: SnapshotAssertion) -> None:
    # Create a DataFrame
    np.random.seed(42)
    table = Table.from_csv_file("C:/Users/ettel/PycharmProjects/Library/tests/resources/_datas/US_Inflation_rates.csv")
    time_series = TimeSeries._from_table(table, target_name="value", time_name="date")
    train_ts, test_ts = time_series.split_rows(0.8)
    model = ArimaModel()

    trained_model = model.fit(train_ts)
    assert snapshot_png == trained_model.predict(test_ts)

