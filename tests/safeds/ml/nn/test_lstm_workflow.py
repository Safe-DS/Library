import pytest
from torch.types import Device

from safeds._config import _get_device
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import RangeScaler
from safeds.ml.nn import (
    ForwardLayer,
    InputConversionTimeSeries,
    LSTMLayer,
    NeuralNetworkRegressor,
    OutputConversionTimeSeries,
)

from tests.helpers import resolve_resource_path, get_devices, get_devices_ids, configure_test_with_device


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
def test_lstm_model(device: Device) -> None:
    configure_test_with_device(device)

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
    assert model._model.state_dict()["_pytorch_layers.0._layer.weight"].device == _get_device()
