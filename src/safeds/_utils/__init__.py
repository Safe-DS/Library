"""Utilities for Safe-DS."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._collections import _compute_duplicates
    from ._hashing import _structural_hash
    from ._lazy import _safe_collect_lazy_frame, _safe_collect_lazy_frame_schema
    from ._plotting import _figure_to_image
    from ._random import _get_random_seed
    from ._string import _get_similar_strings

apipkg.initpkg(
    __name__,
    {
        "_compute_duplicates": "._collections:_compute_duplicates",
        "_figure_to_image": "._plotting:_figure_to_image",
        "_get_random_seed": "._random:_get_random_seed",
        "_get_similar_strings": "._string:_get_similar_strings",
        "_safe_collect_lazy_frame": "._lazy:_safe_collect_lazy_frame",
        "_safe_collect_lazy_frame_schema": "._lazy:_safe_collect_lazy_frame_schema",
        "_structural_hash": "._hashing:_structural_hash",
    },
)

__all__ = [
    "_compute_duplicates",
    "_figure_to_image",
    "_get_random_seed",
    "_get_similar_strings",
    "_safe_collect_lazy_frame",
    "_safe_collect_lazy_frame_schema",
    "_structural_hash",
]
