from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash

from ._column_type import ColumnType

if TYPE_CHECKING:
    import polars as pl


class _PolarsColumnType(ColumnType):
    """
    The type of a column in a table.

    This implementation is based on the data types of polars.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, dtype: pl.DataType):
        self._dtype: pl.DataType = dtype

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _PolarsColumnType):
            return NotImplemented
        if self is other:
            return True
        return self._dtype.is_(other._dtype)

    def __hash__(self) -> int:
        return _structural_hash(self._dtype.__class__.__name__)

    def __repr__(self) -> str:
        return str(self)

    def __sizeof__(self) -> int:
        return self._dtype.__sizeof__()

    def __str__(self) -> str:
        return self._dtype.__str__().split("(", maxsplit=1)[0].lower()

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def is_float(self) -> bool:
        return self._dtype.is_float()

    @property
    def is_int(self) -> bool:
        return self._dtype.is_integer()

    @property
    def is_numeric(self) -> bool:
        return self._dtype.is_numeric()

    @property
    def is_signed_int(self) -> bool:
        return self._dtype.is_signed_integer()

    @property
    def is_temporal(self) -> bool:
        return self._dtype.is_temporal()

    @property
    def is_unsigned_int(self) -> bool:
        return self._dtype.is_unsigned_integer()

    # ------------------------------------------------------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def _polars_data_type(self) -> pl.DataType:
        return self._dtype
