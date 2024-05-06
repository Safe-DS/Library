"""Utilities for Safe-DS."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._file_io import _check_and_normalize_file_path
    from ._hashing import _structural_hash

apipkg.initpkg(
    __name__,
    {
        "_check_and_normalize_file_path": "._file_io:_check_and_normalize_file_path",
        "_structural_hash": "._hashing:_structural_hash",
    },
)

__all__ = [
    "_check_and_normalize_file_path",
    "_structural_hash",
]
