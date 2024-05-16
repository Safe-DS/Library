import pytest
from safeds._config import _get_device
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import RangeScaler
from safeds.ml.nn import (
    NeuralNetworkRegressor,
)
from safeds.ml.nn.converters import (
    InputConversionTimeSeries,
    OutputConversionTimeSeries,
)
from safeds.ml.nn.layers import (
    ForwardLayer,
    LSTMLayer,
)
from torch.types import Device

from tests.helpers import configure_test_with_device, get_devices, get_devices_ids, resolve_resource_path


@pytest.mark.parametrize("device", [get_devices()[0]], ids=[get_devices_ids()[0]])
def test_lstm_model(device: Device) -> None:
    configure_test_with_device(device)

    # Create a DataFrame
    _inflation_path = "_datas/US_Inflation_rates.csv"
    table = Table.from_csv_file(path=resolve_resource_path(_inflation_path))
    table = table.replace_column("date", [table.get_column("date").from_str_to_temporal("%Y-%m-%d")])
    rs = RangeScaler()
    trained_scaler, table = rs.fit_and_transform(table, ["value"])
    train_table, test_table = table.split_rows(0.8, shuffle=False)

    model = NeuralNetworkRegressor(
        InputConversionTimeSeries(window_size=7, forecast_horizon=12, continues= False),
        [ForwardLayer(input_size=7, output_size=256), LSTMLayer(input_size=256, output_size=1)],
        OutputConversionTimeSeries("predicted"),
    )
    model_2 = NeuralNetworkRegressor(
        InputConversionTimeSeries(window_size=7, forecast_horizon=12, continues=True),
        [ForwardLayer(input_size=7, output_size=256), LSTMLayer(input_size=256, output_size=12)],
        OutputConversionTimeSeries("predicted"),
    )
    trained_model = model.fit(train_table.to_time_series_dataset("value", "date"), epoch_size=5)
    trained_model_2 = model_2.fit(train_table.to_time_series_dataset("value", "date"), epoch_size=1)

    pred = trained_model.predict(test_table.to_time_series_dataset("value", "date"))
    pred_list = trained_model_2.predict(test_table.to_time_series_dataset("value", "date"))
    pred_list = pred_list.to_table()
    pred = pred.to_table()
    print(pred)
    print(trained_scaler.inverse_transform_predictions(pred, "value", "predicted"))
    assert False
    assert model._model.state_dict()["_pytorch_layers.0._layer.weight"].device == _get_device()
