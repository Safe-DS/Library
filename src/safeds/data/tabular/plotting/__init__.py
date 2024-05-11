"""Classes that can store tabular data."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._column_plotter import ColumnPlotter
    from ._table_plotter import TablePlotter

apipkg.initpkg(
    __name__,
    {
        "ColumnPlotter": "._column_plotter:ColumnPlotter",
        "TablePlotter": "._table_plotter:TablePlotter",
    },
)

__all__ = [
    "ColumnPlotter",
    "TablePlotter",
]
