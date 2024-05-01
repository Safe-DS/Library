"""Classes that can store labeled data."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._tabular_dataset import TaggedTable

apipkg.initpkg(
    __name__,
    {
        "TaggedTable": "._tabular_dataset:TaggedTable",
    },
)

__all__ = [
    "TaggedTable",
]
