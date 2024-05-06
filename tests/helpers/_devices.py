import pytest
import torch
from torch.types import Device

device_cpu = torch.device("cpu")
device_cuda = torch.device("cuda")


def get_devices() -> list[torch.device]:
    return [device_cpu, device_cuda]


def get_devices_ids() -> list[str]:
    return ["cpu", "cuda"]


def skip_if_device_not_available(device: Device) -> None:
    if device == device_cuda and not torch.cuda.is_available():
        pytest.skip("This test requires cuda")
