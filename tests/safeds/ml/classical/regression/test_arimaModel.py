import pytest
from safeds.ml.classical.regression import ArimaModel
from safeds.data.tabular.containers import TimeSeries, Table, Column
from syrupy import SnapshotAssertion
import pandas as pd
import numpy as np


def test_arimaModel(snapshot_png: SnapshotAssertion) -> None:
    # Create a DataFrame
    table = Table.from_csv_file("C:/Users/ettel/PycharmProjects/Library/tests/resources/_datas/US_Inflation_rates.csv")
    time_series = TimeSeries._from_table(table, target_name="value", time_name="date", feature_names=["feature"])
    tuple_ts = time_series.split_rows()
    model = ArimaModel()

    #right now the model just saves the best parameter for the predict method in the fit method
    trained_model =model.fit(time_series)
    snap = trained_model.predict(time_series)
    assert snapshot_png == snap

