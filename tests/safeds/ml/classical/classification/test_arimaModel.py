import pytest
from safeds.ml.classical.regression import arimaModel
from safeds.data.tabular.containers import TimeSeries, Table
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
    time_series_df = pd.DataFrame({
        'Date': range(length_of_series),
        'Value': synthetic_data,
        'Feature': feature_data  # Add the feature column with random integers
    })
    table = Table._from_pandas_dataframe(time_series_df)
    time_series = TimeSeries._from_table_to_time_series(table, target_name="Value", time_name="Date", feature_names=["Feature"])
    model = arimaModel()
    #right now the model just saves the best parameter for the predict method in the fit method
    trained_model = model.fit(time_series)
    snap = model.predict(time_series)
    assert snapshot_png == snap

