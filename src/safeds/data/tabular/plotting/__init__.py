"""Classes that can store tabular data."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._experimental_column_plotter import ExperimentalColumnPlotter
    from ._experimental_table_plotter import ExperimentalTablePlotter

apipkg.initpkg(
    __name__,
    {
        "ExperimentalColumnPlotter": "._experimental_column_plotter:ExperimentalColumnPlotter",
        "ExperimentalTablePlotter": "._experimental_table_plotter:ExperimentalTablePlotter",
    },
)

__all__ = [
    "ExperimentalColumnPlotter",
    "ExperimentalTablePlotter",
]
