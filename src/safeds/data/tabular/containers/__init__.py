"""Classes that can store tabular data."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._cell import Cell
    from ._column import Column
    from ._row import Row
    from ._string_cell import StringCell
    from ._table import Table
    from ._temporal_cell import TemporalCell

apipkg.initpkg(
    __name__,
    {
        "Cell": "._cell:Cell",
        "Column": "._column:Column",
        "Row": "._row:Row",
        "StringCell": "._string_cell:StringCell",
        "TemporalCell": "._temporal_cell:TemporalCell",
        "Table": "._table:Table",
    },
)

__all__ = [
    "Cell",
    "Column",
    "Row",
    "StringCell",
    "TemporalCell",
    "Table",
]
