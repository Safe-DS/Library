import torch
from torch.types import Device


def _get_device() -> Device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
