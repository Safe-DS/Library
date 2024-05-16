import sys

import pytest
from safeds.data.tabular.containers import Table, Column
from safeds.ml.nn.converters import OutputConversionTimeSeries


def test_output_conversion_time_series() -> None:
    import torch

    with pytest.raises(
        ValueError,
        match=r"The window_size is not set. The data can only be converted if the window_size is provided as `int` in the kwargs.",
    ):
        ot = OutputConversionTimeSeries()
        ot._data_conversion(
            input_data=Table({"a": [1], "c": [1], "b": [1]}).to_time_series_dataset("a", "b"),
            output_data=torch.Tensor([0]),
            win=2,
            kappa=3,
        )


def test_output_conversion_time_series_2() -> None:
    import torch

    with pytest.raises(
        ValueError,
        match=r"The forecast_horizon is not set. The data can only be converted if the forecast_horizon is provided as `int` in the kwargs.",
    ):
        ot = OutputConversionTimeSeries()
        ot._data_conversion(
            input_data=Table({"a": [1], "c": [1], "b": [1]}).to_time_series_dataset("a", "b"),
            output_data=torch.Tensor([0]),
            window_size=2,
            kappa=3,
        )


class TestEq:
    @pytest.mark.parametrize(
        ("output_conversion_ts1", "output_conversion_ts2"),
        [
            (OutputConversionTimeSeries(), OutputConversionTimeSeries()),
            (OutputConversionTimeSeries(), OutputConversionTimeSeries()),
            (OutputConversionTimeSeries(), OutputConversionTimeSeries()),
        ],
    )
    def test_should_be_equal(
        self,
        output_conversion_ts1: OutputConversionTimeSeries,
        output_conversion_ts2: OutputConversionTimeSeries,
    ) -> None:
        assert output_conversion_ts1 == output_conversion_ts2

    @pytest.mark.parametrize(
        ("output_conversion_ts1", "output_conversion_ts2"),
        [
            (OutputConversionTimeSeries(), Table()),
            (OutputConversionTimeSeries("2"), Column("test")),
        ],
    )
    def test_should_not_be_equal(
        self,
        output_conversion_ts1: OutputConversionTimeSeries,
        output_conversion_ts2: OutputConversionTimeSeries,
    ) -> None:
        assert output_conversion_ts1 != output_conversion_ts2


class TestHash:
    @pytest.mark.parametrize(
        ("output_conversion_ts1", "output_conversion_ts2"),
        [
            (OutputConversionTimeSeries(), OutputConversionTimeSeries()),
            (OutputConversionTimeSeries(), OutputConversionTimeSeries()),
            (OutputConversionTimeSeries(), OutputConversionTimeSeries()),
        ],
    )
    def test_hash_should_be_equal(
        self,
        output_conversion_ts1: OutputConversionTimeSeries,
        output_conversion_ts2: OutputConversionTimeSeries,
    ) -> None:
        assert hash(output_conversion_ts1) == hash(output_conversion_ts2)


class TestSizeOf:
    @pytest.mark.parametrize(
        "output_conversion_ts",
        [
            OutputConversionTimeSeries("1"),
            OutputConversionTimeSeries("2"),
            OutputConversionTimeSeries("3"),
        ],
    )
    def test_should_size_be_greater_than_normal_object(
        self,
        output_conversion_ts: OutputConversionTimeSeries,
    ) -> None:
        assert sys.getsizeof(output_conversion_ts) > sys.getsizeof(object())
