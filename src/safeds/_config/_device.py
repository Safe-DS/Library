from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.types import Device


def _get_device() -> Device:
    import torch

    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
