"""Classes that can store labeled data."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._tagged_table import TaggedTable

apipkg.initpkg(
    __name__,
    {
        "TaggedTable": "._tagged_table:TaggedTable",
    },
)

__all__ = [
    "TaggedTable",
]
