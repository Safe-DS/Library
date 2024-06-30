import pytest
from safeds._config import _get_device
from safeds.data.tabular.containers import Table
from safeds.ml.nn import (
    NeuralNetworkRegressor,
)
from safeds.ml.nn.converters import (
    InputConversionTable,
)
from safeds.ml.nn.layers import (
    DropoutLayer,
    ForwardLayer,
)
from torch.types import Device

from tests.helpers import configure_test_with_device, get_devices, get_devices_ids


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
def test_forward_model(device: Device) -> None:
    configure_test_with_device(device)

    train_table = Table(
        {
            "feature": [1, 2, 3],
            "value": [1, 2, 3],
        },
    )

    model = NeuralNetworkRegressor(
        InputConversionTable(),
        [ForwardLayer(neuron_count=1), DropoutLayer(probability=0.5)],
    )

    fitted_model = model.fit(train_table.to_tabular_dataset("value"), epoch_size=1, learning_rate=0.01)
    assert fitted_model._model is not None
    assert fitted_model._model.state_dict()["_pytorch_layers.0._layer.weight"].device == _get_device()
