"""Classes that represent queries on the data."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._string_cell import StringCell
    from ._temporal_cell import TemporalCell

apipkg.initpkg(
    __name__,
    {
        "StringCell": "._string_cell:StringCell",
        "TemporalCell": "._temporal_cell:TemporalCell",
    },
)

__all__ = [
    "StringCell",
    "TemporalCell",
]
