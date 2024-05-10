from __future__ import annotations

from typing import TYPE_CHECKING

from ._data_type import DataType

if TYPE_CHECKING:
    import polars as pl


class _PolarsDataType(DataType):
    """
    The type of a column or cell in a table.

    This implementation is based on Polars' data types.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, dtype: pl.DataType):
        self._dtype: pl.DataType = dtype

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _PolarsDataType):
            return NotImplemented
        if self is other:
            return True
        return self._dtype.is_(other._dtype)

    def __hash__(self) -> int:
        return self._dtype.__hash__()

    def __repr__(self) -> str:
        return self._dtype.__repr__()

    def __sizeof__(self) -> int:
        return self._dtype.__sizeof__()

    def __str__(self) -> str:
        return self._dtype.__str__()

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def is_numeric(self) -> bool:
        return self._dtype.is_numeric()

    @property
    def is_temporal(self) -> bool:
        return self._dtype.is_temporal()
