"""Configuration for Safe-DS."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._torch import _get_device, _init_default_device, _set_default_device

apipkg.initpkg(
    __name__,
    {
        "_get_device": "._torch:_get_device",
        "_init_default_device": "._torch:_init_default_device",
        "_set_default_device": "._torch:_set_default_device",
    },
)

__all__ = [
    "_get_device",
    "_init_default_device",
    "_set_default_device",
]
