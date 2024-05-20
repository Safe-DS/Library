import pytest
from safeds._config import _get_device
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import StandardScaler
from safeds.ml.nn import (
    NeuralNetworkRegressor,
)
from safeds.ml.nn.layers import (
    ForwardLayer,
)
from torch.types import Device

from tests.helpers import configure_test_with_device, get_devices, get_devices_ids, resolve_resource_path


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
def test_forward_model(device: Device) -> None:
    configure_test_with_device(device)

    # Create a DataFrame
    _inflation_path = "_datas/US_Inflation_rates.csv"
    table_1 = Table.from_csv_file(
        path=resolve_resource_path(_inflation_path),
    )
    table_1 = table_1.remove_columns(["date"])
    table_2 = table_1.slice_rows(start=0, length=table_1.row_count - 14)
    table_2 = table_2.add_columns([(table_1.slice_rows(start=14)).get_column("value").rename("target")])
    train_table, test_table = table_2.split_rows(0.8)

    ss = StandardScaler()
    _, train_table = ss.fit_and_transform(train_table, ["value"])
    _, test_table = ss.fit_and_transform(test_table, ["value"])
    model = NeuralNetworkRegressor(
        [ForwardLayer(input_size=1, output_size=1)],
    )

    fitted_model = model.fit(train_table.to_tabular_dataset("target"), epoch_size=1, learning_rate=0.01)
    fitted_model.predict(test_table.remove_columns_except(["value"]))
    assert fitted_model._model is not None
    assert fitted_model._model.state_dict()["_pytorch_layers.0._layer.weight"].device == _get_device()
