from __future__ import annotations

from typing import TYPE_CHECKING

from safeds.exceptions import UnknownColumnNameError

from ._experimental_lazy_cell import _LazyCell
from ._experimental_row import ExperimentalRow

if TYPE_CHECKING:
    from safeds.data.tabular.typing import ExperimentalSchema
    from safeds.data.tabular.typing._experimental_data_type import ExperimentalDataType

    from ._experimental_table import ExperimentalTable


class _LazyVectorizedRow(ExperimentalRow):
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

    def __init__(self, table: ExperimentalTable):
        self._table: ExperimentalTable = table

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
    def number_of_columns(self) -> int:
        return self._table.number_of_columns

    @property
    def schema(self) -> ExperimentalSchema:
        return self._table.schema

    # ------------------------------------------------------------------------------------------------------------------
    # Column operations
    # ------------------------------------------------------------------------------------------------------------------

    def get_value(self, name: str) -> _LazyCell:
        import polars as pl

        if not self._table.has_column(name):
            raise UnknownColumnNameError([name])

        return _LazyCell(pl.col(name))

    def get_column_type(self, name: str) -> ExperimentalDataType:
        return self._table.get_column_type(name)

    def has_column(self, name: str) -> bool:
        return self._table.has_column(name)
