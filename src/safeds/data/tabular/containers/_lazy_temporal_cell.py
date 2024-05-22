from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _ClosedBound

from ._lazy_cell import _LazyCell
from ._temporal_cell import TemporalCell

if TYPE_CHECKING:
    import datetime

    import polars as pl

    from ._cell import Cell


class _LazyTemporalCell(TemporalCell):
    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, expression: pl.Expr) -> None:
        self._expression: pl.Expr = expression

    def __hash__(self) -> int:
        return _structural_hash(self._expression.meta.serialize())

    def __sizeof__(self) -> int:
        return self._expression.__sizeof__()

    # ------------------------------------------------------------------------------------------------------------------
    # Temporal operations
    # ------------------------------------------------------------------------------------------------------------------

    def date_to_string(self) -> Cell[str | None]:
        return _LazyCell(self._expression.dt.to_string(format="%F"))

    def datetime_to_string(self) -> Cell[str | None]:

        return _LazyCell(self._expression.dt.to_string(format="%Y/%m/%d %H:%M:%S"))
    # ------------------------------------------------------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------------------------------------------------------

    def _equals(self, other: object) -> bool:
        if not isinstance(other, _LazyTemporalCell):
            return NotImplemented
        if self is other:
            return True
        return self._expression.meta.eq(other._expression.meta)
