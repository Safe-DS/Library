"""Utilities for Safe-DS."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._file_io import _check_and_normalize_file_path
    from ._hashing import _structural_hash
    from ._plotting import _figure_to_image
    from ._random import _get_random_seed

apipkg.initpkg(
    __name__,
    {
        "_check_and_normalize_file_path": "._file_io:_check_and_normalize_file_path",
        "_structural_hash": "._hashing:_structural_hash",
        "_figure_to_image": "._plotting:_figure_to_image",
        "_get_random_seed": "._random:_get_random_seed",
    },
)

__all__ = [
    "_check_and_normalize_file_path",
    "_structural_hash",
    "_figure_to_image",
    "_get_random_seed",
]
