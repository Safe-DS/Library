import pytest
import torch
from safeds._config import _init_default_device, _set_default_device
from torch.types import Device

_init_default_device()

device_cpu = torch.device("cpu")
device_cuda = torch.device("cuda")


def get_devices() -> list[torch.device]:
    return [device_cpu, device_cuda]


def get_devices_ids() -> list[str]:
    return ["cpu", "cuda"]


def configure_test_with_device(device: Device) -> None:
    _skip_if_device_not_available(device)  # This will end the function if device is not available
    _set_default_device(device)


def _skip_if_device_not_available(device: Device) -> None:
    if device == device_cuda and not torch.cuda.is_available():
        pytest.skip("This test requires cuda")
