"""Classes that can store tabular data."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._column import Column
    from ._experimental_cell import ExperimentalCell

apipkg.initpkg(
    __name__,
    {
        "Column": "._column:Column",
        "ExperimentalCell": "._experimental_cell:ExperimentalCell",
    },
)

__all__ = [
    "Column",
    "ExperimentalCell",
]
