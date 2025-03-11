from __future__ import annotations
from typing import TYPE_CHECKING


from src.safeds._utils import _structural_hash

from src.safeds.data.tabular.query._column_selector import ColumnSelector


if TYPE_CHECKING:
    from polars._typing import SelectorType


class _LazyColumnSelector(ColumnSelector):
    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, expression: SelectorType) -> None:
        self._selector: SelectorType = expression

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _LazyColumnSelector):
            return NotImplemented
        if self is other:
            return True
        return self._selector.meta.eq(other._selector)

    def __hash__(self) -> int:
        return _structural_hash(self._selector.meta.serialize())

    def __repr__(self) -> str:
        return f"_LazyColumnSelector({self._selector})"

    def __sizeof__(self) -> int:
        return self._selector.__sizeof__()

    def __str__(self) -> str:
        return self._selector.__str__()
