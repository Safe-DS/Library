import pytest
from safeds.ml.classical.regression import ArimaModel
from safeds.data.tabular.containers import TimeSeries, Table, Column
from syrupy import SnapshotAssertion
import pandas as pd
import numpy as np


def test_arimaModel(snapshot_png: SnapshotAssertion) -> None:
    # Define the length of the time series
    length_of_series = 100

    # Generate synthetic data using a random normal distribution
    synthetic_data = np.random.randn(length_of_series)

    # Generate a 'Feature' column with random integers
    # For example, create random integers between 0 and 100
    feature_data = np.random.randint(0, 100, length_of_series)

    # Create a DataFrame
    table = Table.from_csv_file("C:/Users/ettel/PycharmProjects/Library/tests/resources/_datas/US_Inflation_rates.csv")
    col = Column("feature", range(0,918))
    table = table.add_column(col)
    time_series = TimeSeries._from_table_to_time_series(table, target_name="value", time_name="date", feature_names=["feature"])
    tuple_ts = time_series.split_rows()
    model = ArimaModel()
    #right now the model just saves the best parameter for the predict method in the fit method
    trained_model =model.fit(time_series)
    snap = trained_model.predict(time_series)
    assert snapshot_png == snap

