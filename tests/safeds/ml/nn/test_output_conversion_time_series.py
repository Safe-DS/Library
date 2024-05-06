import pytest
from safeds.data.tabular.containers import Table
from safeds.ml.nn import OutputConversionTimeSeries


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
