from safeds.data.tabular.containers import Table
from safeds.ml.nn import (
    NeuralNetworkRegressor,
)
from safeds.ml.nn.converters import (
    InputConversionTimeSeries,
    OutputConversionTimeSeries,
)
from safeds.ml.nn.layers import (
    LSTMLayer,
)


def test_should_raise_if_is_fitted_is_set_correctly_lstm() -> None:
    model = NeuralNetworkRegressor(
        InputConversionTimeSeries(1, 1),
        [LSTMLayer(input_size=2, output_size=1)],
        OutputConversionTimeSeries("predicted"),
    )
    ts = Table.from_dict({"target": [1, 1, 1, 1], "time": [0, 0, 0, 0], "feat": [0, 0, 0, 0]}).to_time_series_dataset(
        "target",
        "time",
    )
    assert not model.is_fitted
    model = model.fit(ts)
    model.predict(ts)
    assert model.is_fitted


def test_get_output_config() -> None:
    test_val = {"window_size": 1, "forecast_horizon": 1}
    it = InputConversionTimeSeries(1, 1)
    di = it._get_output_configuration()
    assert di == test_val
