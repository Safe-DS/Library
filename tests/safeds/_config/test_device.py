import pytest
import torch
from torch.types import Device

from safeds._config import _get_device
from safeds._config._device import _set_default_device
from tests.helpers import get_devices, get_devices_ids, configure_test_with_device


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
def test_default_device(device: Device) -> None:
    configure_test_with_device(device)
    assert _get_device().type == device.type
    assert torch.get_default_device().type == device.type


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
def test_set_default_device(device: Device) -> None:
    _set_default_device(device)
    assert _get_device().type == device.type
    assert torch.get_default_device().type == device.type
