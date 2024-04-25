from safeds.data.tabular.containers import Table
from safeds.ml.nn import (
    InputConversionTimeSeries,
    LSTMLayer,
    NeuralNetworkRegressor,
    OutputConversionTimeSeries,
)


def test_should_raise_if_is_fitted_is_set_correctly_lstm() -> None:
    model = NeuralNetworkRegressor(
        InputConversionTimeSeries(1, 1, "target", "time", ["feat"]),
        [LSTMLayer(input_size=2, output_size=1)],
        OutputConversionTimeSeries("predicted"),
    )
    ts = Table.from_dict({"target": [1, 1, 1, 1], "time": [0, 0, 0, 0], "feat": [0, 0, 0, 0]}).time_columns(
        "target", "time", ["feat"],
    )
    assert not model.is_fitted
    model = model.fit(ts)
    model.predict(ts)
    assert model.is_fitted
