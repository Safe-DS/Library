"""Classes that can store tabular data."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from _exerimental_polars_row import ExperimentalPolarsRow

    from ._column import Column
    from ._experimental_polars_cell import ExperimentalPolarsCell
    from ._experimental_polars_column import ExperimentalPolarsColumn
    from ._experimental_polars_table import ExperimentalPolarsTable
    from ._row import Row
    from ._table import Table
    from ._time_series import TimeSeries

apipkg.initpkg(
    __name__,
    {
        "Column": "._column:Column",
        "ExperimentalPolarsCell": "._experimental_polars_cell:ExperimentalPolarsCell",
        "ExperimentalPolarsColumn": "._experimental_polars_column:ExperimentalPolarsColumn",
        "ExperimentalPolarsRow": "._experimental_polars_row:ExperimentalPolarsRow",
        "ExperimentalPolarsTable": "._experimental_polars_table:ExperimentalPolarsTable",
        "Row": "._row:Row",
        "Table": "._table:Table",
        "TimeSeries": "._time_series:TimeSeries",
    },
)

__all__ = [
    "Column",
    "ExperimentalPolarsCell",
    "ExperimentalPolarsColumn",
    "ExperimentalPolarsRow",
    "ExperimentalPolarsTable",
    "Row",
    "Table",
    "TimeSeries",
]
