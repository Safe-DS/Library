import sys

import pytest
from safeds.data.tabular.containers import Table
from safeds.ml.nn import (
    NeuralNetworkRegressor,
)
from safeds.ml.nn.converters import (
    InputConversionTimeSeries,
)
from safeds.ml.nn.layers import (
    LSTMLayer,
)


def test_should_raise_if_is_fitted_is_set_correctly_lstm() -> None:
    model = NeuralNetworkRegressor(
        InputConversionTimeSeries(),
        [LSTMLayer(neuron_count=1)],
    )
    ts = Table.from_dict({"target": [1, 1, 1, 1], "time": [0, 0, 0, 0], "feat": [0, 0, 0, 0]}).to_time_series_dataset(
        target_name="target",
        window_size=1,
    )
    assert not model.is_fitted
    model = model.fit(ts)
    model.predict(ts.to_table())
    assert model.is_fitted


def test_is_predict_data_valid() -> None:
    input_conv = InputConversionTimeSeries()
    data = Table({"target": [1, 1, 1, 1], "time": [0, 0, 0, 0], "feat": [0, 0, 0, 0]})
    assert not input_conv._is_predict_data_valid(data)
    input_conv._feature_names = ["XYZ"]
    assert not input_conv._is_predict_data_valid(data)


class TestEq:
    @pytest.mark.parametrize(
        ("output_conversion_ts1", "output_conversion_ts2"),
        [
            (InputConversionTimeSeries(), InputConversionTimeSeries()),
        ],
    )
    def test_should_be_equal(
        self,
        output_conversion_ts1: InputConversionTimeSeries,
        output_conversion_ts2: InputConversionTimeSeries,
    ) -> None:
        assert output_conversion_ts1 == output_conversion_ts2

    @pytest.mark.parametrize(
        ("output_conversion_ts1", "output_conversion_ts2"),
        [
            (
                InputConversionTimeSeries(),
                Table(),
            ),
        ],
    )
    def test_should_not_be_equal(
        self,
        output_conversion_ts1: InputConversionTimeSeries,
        output_conversion_ts2: InputConversionTimeSeries,
    ) -> None:
        assert output_conversion_ts1 != output_conversion_ts2


class TestHash:
    @pytest.mark.parametrize(
        ("output_conversion_ts1", "output_conversion_ts2"),
        [
            (InputConversionTimeSeries(), InputConversionTimeSeries()),
        ],
    )
    def test_hash_should_be_equal(
        self,
        output_conversion_ts1: InputConversionTimeSeries,
        output_conversion_ts2: InputConversionTimeSeries,
    ) -> None:
        assert hash(output_conversion_ts1) == hash(output_conversion_ts2)

    def test_hash_should_not_be_equal(self) -> None:
        output_conversion_ts1 = InputConversionTimeSeries()
        output_conversion_ts2 = InputConversionTimeSeries()
        output_conversion_ts3 = InputConversionTimeSeries()
        assert hash(output_conversion_ts1) == hash(output_conversion_ts3)
        assert hash(output_conversion_ts2) == hash(output_conversion_ts1)
        assert hash(output_conversion_ts3) == hash(output_conversion_ts2)


class TestSizeOf:
    @pytest.mark.parametrize(
        "output_conversion_ts",
        [
            InputConversionTimeSeries(),
        ],
    )
    def test_should_size_be_greater_than_normal_object(
        self,
        output_conversion_ts: InputConversionTimeSeries,
    ) -> None:
        assert sys.getsizeof(output_conversion_ts) > sys.getsizeof(object())
