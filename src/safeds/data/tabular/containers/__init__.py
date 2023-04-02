"""Classes that can store tabular data."""

from ._column import Column
from ._row import Row
from ._table import Table
from ._tagged_table import TaggedTable

__all__ = [
    "Column",
    "Row",
    "Table",
    "TaggedTable",
]
