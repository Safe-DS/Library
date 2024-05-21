import sys

import pytest
from safeds.data.tabular.containers import Table
from safeds.ml.nn import (
    NeuralNetworkRegressor,
)
from safeds.ml.nn._converters import (
    _TimeSeriesConverter,
)
from safeds.ml.nn.layers import (
    LSTMLayer,
)


def test_should_raise_if_is_fitted_is_set_correctly_lstm() -> None:
    model = NeuralNetworkRegressor(
        _TimeSeriesConverter(prediction_name="predicted"),
        [LSTMLayer(input_size=2, output_size=1)],
    )
    ts = Table.from_dict({"target": [1, 1, 1, 1], "time": [0, 0, 0, 0], "feat": [0, 0, 0, 0]}).to_time_series_dataset(
        target_name="target",
        time_name="time",
        window_size=1,
    )
    assert not model.is_fitted
    model = model.fit(ts)
    model.predict(ts)
    assert model.is_fitted


class TestEq:
    @pytest.mark.parametrize(
        ("output_conversion_ts1", "output_conversion_ts2"),
        [
            (_TimeSeriesConverter(), _TimeSeriesConverter()),
        ],
    )
    def test_should_be_equal(
        self,
        output_conversion_ts1: _TimeSeriesConverter,
        output_conversion_ts2: _TimeSeriesConverter,
    ) -> None:
        assert output_conversion_ts1 == output_conversion_ts2

    @pytest.mark.parametrize(
        ("output_conversion_ts1", "output_conversion_ts2"),
        [
            (
                _TimeSeriesConverter(),
                Table(),
            ),
            (
                _TimeSeriesConverter(prediction_name="2"),
                _TimeSeriesConverter(prediction_name="1"),
            ),
        ],
    )
    def test_should_not_be_equal(
        self,
        output_conversion_ts1: _TimeSeriesConverter,
        output_conversion_ts2: _TimeSeriesConverter,
    ) -> None:
        assert output_conversion_ts1 != output_conversion_ts2


class TestHash:
    @pytest.mark.parametrize(
        ("output_conversion_ts1", "output_conversion_ts2"),
        [
            (_TimeSeriesConverter(), _TimeSeriesConverter()),
        ],
    )
    def test_hash_should_be_equal(
        self,
        output_conversion_ts1: _TimeSeriesConverter,
        output_conversion_ts2: _TimeSeriesConverter,
    ) -> None:
        assert hash(output_conversion_ts1) == hash(output_conversion_ts2)

    def test_hash_should_not_be_equal(self) -> None:
        output_conversion_ts1 = _TimeSeriesConverter(prediction_name="1")
        output_conversion_ts2 = _TimeSeriesConverter(prediction_name="2")
        output_conversion_ts3 = _TimeSeriesConverter(prediction_name="3")
        assert hash(output_conversion_ts1) != hash(output_conversion_ts3)
        assert hash(output_conversion_ts2) != hash(output_conversion_ts1)
        assert hash(output_conversion_ts3) != hash(output_conversion_ts2)


class TestSizeOf:
    @pytest.mark.parametrize(
        "output_conversion_ts",
        [
            _TimeSeriesConverter(prediction_name="1"),
            _TimeSeriesConverter(prediction_name="2"),
            _TimeSeriesConverter(prediction_name="3"),
        ],
    )
    def test_should_size_be_greater_than_normal_object(
        self,
        output_conversion_ts: _TimeSeriesConverter,
    ) -> None:
        assert sys.getsizeof(output_conversion_ts) > sys.getsizeof(object())
