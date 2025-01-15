"""Classes that represent queries on the data."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._string_operations import StringOperations
    from ._temporal_operations import TemporalOperations

apipkg.initpkg(
    __name__,
    {
        "StringOperations": "._string_operations:StringOperations",
        "TemporalOperations": "._temporal_operations:TemporalOperations",
    },
)

__all__ = [
    "StringOperations",
    "TemporalOperations",
]
