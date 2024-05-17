"""Classes that can store tabular data."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._temporal import Temporal

apipkg.initpkg(
    __name__,
    {
        "Temporal": "._temporal:Temporal",
    },
)

__all__ = [
    "Temporal",
]
