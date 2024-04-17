"""Utilities for Safe-DS."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._hashing import _structural_hash

apipkg.initpkg(
    __name__,
    {
        "_structural_hash": "._hashing:_structural_hash",
    },
)

__all__ = [
    "_structural_hash",
]
