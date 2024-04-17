"""Configuration for Safe-DS."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._device import _get_device

apipkg.initpkg(
    __name__,
    {
        "_get_device": "._device:_get_device",
    },
)

__all__ = [
    "_get_device",
]
