import pytest
import torch
from torch.types import Device

from safeds._config import _get_device, _init_default_device
from safeds._config._device import _set_default_device
from tests.helpers import get_devices, get_devices_ids, configure_test_with_device, device_cuda, device_cpu
from tests.helpers._devices import _skip_if_device_not_available


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
def test_default_device(device: Device) -> None:
    configure_test_with_device(device)
    assert _get_device().type == device.type
    assert torch.get_default_device().type == device.type


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
def test_set_default_device(device: Device) -> None:
    _skip_if_device_not_available(device)
    _set_default_device(device)
    assert _get_device().type == device.type
    assert torch.get_default_device().type == device.type


def test_init_default_device() -> None:
    _init_default_device()
    if torch.cuda.is_available():
        assert _get_device().type == device_cuda.type
        assert torch.get_default_device().type == device_cuda.type
    else:
        assert _get_device().type == device_cpu.type
        assert torch.get_default_device().type == device_cpu.type
