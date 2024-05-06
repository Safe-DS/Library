from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import RangeScaler
from safeds.ml.nn import (
    ForwardLayer,
    InputConversionTimeSeries,
    LSTMLayer,
    NeuralNetworkRegressor,
    OutputConversionTimeSeries,
)

from tests.helpers import resolve_resource_path


def test_lstm_model() -> None:
    # Create a DataFrame
    _inflation_path = "_datas/US_Inflation_rates.csv"
    table = Table.from_csv_file(path=resolve_resource_path(_inflation_path))
    rs = RangeScaler()
    _, table = rs.fit_and_transform(table, ["value"])
    train_table, test_table = table.split_rows(0.8)

    model = NeuralNetworkRegressor(
        InputConversionTimeSeries(window_size=7, forecast_horizon=12),
        [ForwardLayer(input_size=7, output_size=256), LSTMLayer(input_size=256, output_size=1)],
        OutputConversionTimeSeries("predicted"),
    )
    trained_model = model.fit(train_table.to_time_series_dataset("value", "date"), epoch_size=1)

    trained_model.predict(test_table.to_time_series_dataset("value", "date"))

