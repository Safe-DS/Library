"""Classes that can store tabular data."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._column import Column
    from ._row import Row
    from ._table import Table
    from ._time_series import TimeSeries

apipkg.initpkg(
    __name__,
    {
        "Column": "._column:Column",
        "Row": "._row:Row",
        "Table": "._table:Table",
        "TimeSeries": "._time_series:TimeSeries",
    },
)

__all__ = [
    "Column",
    "Row",
    "Table",
    "TimeSeries",
]
