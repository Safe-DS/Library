import pytest
import torch
from safeds._config import _get_device, _init_default_device, _set_default_device
from torch.types import Device

from tests.helpers import configure_test_with_device, device_cpu, device_cuda, get_devices, get_devices_ids
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
