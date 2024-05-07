"""Classes that can store tabular data."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._column import Column
    from ._experimental_cell import ExperimentalCell
    from ._experimental_column import ExperimentalColumn
    from ._experimental_row import ExperimentalRow
    from ._experimental_table import ExperimentalTable
    from ._row import Row
    from ._table import Table

apipkg.initpkg(
    __name__,
    {
        "Column": "._column:Column",
        "ExperimentalCell": "._experimental_cell:ExperimentalCell",
        "ExperimentalColumn": "._experimental_column:ExperimentalColumn",
        "ExperimentalRow": "._experimental_row:ExperimentalRow",
        "ExperimentalTable": "._experimental_table:ExperimentalTable",
        "Row": "._row:Row",
        "Table": "._table:Table",
    },
)

__all__ = [
    "Column",
    "ExperimentalCell",
    "ExperimentalColumn",
    "ExperimentalRow",
    "ExperimentalTable",
    "Row",
    "Table",
]
