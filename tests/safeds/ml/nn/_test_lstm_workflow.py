from typing import Any

import pytest
from safeds.data.tabular.containers import TimeSeries

from safeds.ml.nn import (
    ForwardLayer,
    InputConversionTimeSeries,
    NeuralNetworkRegressor,
    OutputConversionTimeSeries,
    LSTMLayer,
)


from tests.helpers import resolve_resource_path


def test_lstm_model() -> None:
    # Create a DataFrame
    _inflation_path = "_datas/US_Inflation_rates.csv"
    time_series = TimeSeries.timeseries_from_csv_file(
        path=resolve_resource_path(_inflation_path),
        target_name="value",
        time_name="date",
    )
    train_ts, test_ts = time_series.split_rows(0.8)
    model = NeuralNetworkRegressor(
        InputConversionTimeSeries(window_size=7, forecast_horizon=12, time_name="date", target_name="value"),
        [ForwardLayer(input_size=7, output_size=1)],
        OutputConversionTimeSeries("predicted"),
    )
    trained_model = model.fit(train_ts)
    predictions = trained_model.predict(test_ts)
    print(predictions)
    # suggest it ran through
    assert False
