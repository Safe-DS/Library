from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.types import Device


_default_device: Device | None = None


def _get_device() -> Device:
    import torch

    return torch.get_default_device()


def _init_default_device() -> None:
    import torch

    global _default_device  # noqa: PLW0603

    if _default_device is None:
        _default_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    torch.set_default_device(_default_device)


def _set_default_device(device: Device) -> None:
    # This changes all future tensors, but not any tensor that already exists
    global _default_device  # noqa: PLW0603

    _default_device = device
    _init_default_device()
