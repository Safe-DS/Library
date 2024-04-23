import pytest
from safeds.data.tabular.containers import Table, TaggedTable, TimeSeries
from safeds.exceptions import FeatureDataMismatchError, InputSizeError, ModelNotFittedError, OutOfBoundsError
from safeds.ml.nn import (
    InputConversionTimeSeries,
    NeuralNetworkRegressor,
    OutputConversionTimeSeries,
    LSTMLayer,
)


def test_should_raise_if_is_fitted_is_set_correctly_lstm() -> None:
    model = NeuralNetworkRegressor(
        InputConversionTimeSeries(1, 1, "target", "time"),
        [LSTMLayer(input_size=1, output_size=1)],
        OutputConversionTimeSeries("predicted"),
    )
    assert not model.is_fitted
    model = model.fit(
        Table.from_dict({"target": [1, 1, 1, 1], "time": [0, 0, 0, 0], "feat": [0, 0, 0, 0]}).time_columns("target",
                                                                                                           "time"),
    )
    assert model.is_fitted
