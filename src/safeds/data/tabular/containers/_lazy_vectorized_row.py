from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._validation import _check_columns_exist

from ._lazy_cell import _LazyCell
from ._row import Row

if TYPE_CHECKING:
    from safeds.data.tabular.typing import DataType, Schema

    from ._table import Table


class _LazyVectorizedRow(Row):
    """
    A one-dimensional collection of named, heterogeneous values.

    This implementation treats an entire table as a row, where each column is a "cell" in the row. This greatly speeds
    up operations on the row.

    Moreover, accessing a column only builds an expression that will be evaluated when needed. This is useful when later
    operations remove more rows or columns, so we don't do unnecessary work upfront.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, table: Table):
        self._table: Table = table

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _LazyVectorizedRow):
            return NotImplemented
        if self is other:
            return True
        return self._table == other._table

    def __hash__(self) -> int:
        return self._table.__hash__()

    def __sizeof__(self) -> int:
        return self._table.__sizeof__()

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def column_names(self) -> list[str]:
        return self._table.column_names

    @property
    def column_count(self) -> int:
        return self._table.column_count

    @property
    def schema(self) -> Schema:
        return self._table.schema

    # ------------------------------------------------------------------------------------------------------------------
    # Column operations
    # ------------------------------------------------------------------------------------------------------------------

    def get_value(self, name: str) -> _LazyCell:
        import polars as pl

        _check_columns_exist(self._table, name)

        return _LazyCell(pl.col(name))

    def get_column_type(self, name: str) -> DataType:
        return self._table.get_column_type(name)

    def has_column(self, name: str) -> bool:
        return self._table.has_column(name)
