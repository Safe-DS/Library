from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash

from ._column_selector import ColumnSelector

if TYPE_CHECKING:
    from polars._typing import SelectorType


class _LazyColumnSelector(ColumnSelector):
    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, selector: SelectorType) -> None:
        self._selector: SelectorType = selector

    def __invert__(self) -> ColumnSelector:
        return _LazyColumnSelector(self._selector.__invert__())

    def __and__(self, other: ColumnSelector) -> ColumnSelector:
        return _LazyColumnSelector(self._selector.__and__(other._polars_selector))

    def __rand__(self, other: ColumnSelector) -> ColumnSelector:
        expression = self._selector.__rand__(other._polars_selector).meta._as_selector()
        return _LazyColumnSelector(expression)

    def __or__(self, other: ColumnSelector) -> ColumnSelector:
        return _LazyColumnSelector(self._selector.__or__(other._polars_selector))

    def __ror__(self, other: ColumnSelector) -> ColumnSelector:
        expression = self._selector.__ror__(other._polars_selector).meta._as_selector()
        return _LazyColumnSelector(expression)

    def __sub__(self, other: ColumnSelector) -> ColumnSelector:
        return _LazyColumnSelector(self._selector.__sub__(other._polars_selector))

    def __rsub__(self, other: ColumnSelector) -> ColumnSelector:
        expression = self._selector.__rsub__(other._polars_selector).meta._as_selector()
        return _LazyColumnSelector(expression)

    def __xor__(self, other: ColumnSelector) -> ColumnSelector:
        return _LazyColumnSelector(self._selector.__xor__(other._polars_selector))

    def __rxor__(self, other: ColumnSelector) -> ColumnSelector:
        expression = self._selector.__rxor__(other._polars_selector).meta._as_selector()
        return _LazyColumnSelector(expression)

    # Other --------------------------------------------------------------------

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

    # ------------------------------------------------------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def _polars_selector(self) -> SelectorType:
        return self._selector
