import pytest
from safeds._config import _get_device
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import RangeScaler
from safeds.ml.nn import (
    NeuralNetworkRegressor,
)
from safeds.ml.nn.converters import (
    InputConversionTimeSeries,
)
from safeds.ml.nn.layers import (
    ForwardLayer,
    LSTMLayer,
)
from torch.types import Device

from tests.helpers import configure_test_with_device, get_devices, get_devices_ids, resolve_resource_path


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
def test_lstm_model(device: Device) -> None:
    configure_test_with_device(device)

    # Create a DataFrame
    _inflation_path = "_datas/US_Inflation_rates.csv"
    table = Table.from_csv_file(path=resolve_resource_path(_inflation_path))
    rs = RangeScaler(column_names="value")
    _, table = rs.fit_and_transform(table)
    train_table, test_table = table.split_rows(0.8)

    model = NeuralNetworkRegressor(
        InputConversionTimeSeries(),
        [ForwardLayer(input_size=7, output_size=256), LSTMLayer(input_size=256, output_size=1)],
    )
    model_2 = NeuralNetworkRegressor(
        InputConversionTimeSeries(),
        [ForwardLayer(input_size=7, output_size=256), LSTMLayer(input_size=256, output_size=12)],
    )
    trained_model_2 = model_2.fit(
        train_table.to_time_series_dataset(
            "value",
            "date",
            window_size=7,
            forecast_horizon=12,
            continuous=True,
        ),
        epoch_size=1,
    )
    trained_model_2.predict(
        test_table.to_time_series_dataset(
            "value",
            "date",
            window_size=7,
            forecast_horizon=12,
            continuous=True,
        ),
    )
    trained_model = model.fit(
        train_table.to_time_series_dataset(
            "value",
            "date",
            window_size=7,
            forecast_horizon=12,
            continuous=False,
        ),
        epoch_size=1,
    )

    trained_model.predict(
        test_table.to_time_series_dataset("value", "date", window_size=7, forecast_horizon=12, continuous=False),
    )
    assert trained_model._model is not None
    assert trained_model._model.state_dict()["_pytorch_layers.0._layer.weight"].device == _get_device()
