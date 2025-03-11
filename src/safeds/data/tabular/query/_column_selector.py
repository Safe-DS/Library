from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

    from safeds.data.tabular.typing import ColumnType


class ColumnSelector:
    # ------------------------------------------------------------------------------------------------------------------
    # Static methods
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def all() -> ColumnSelector:
        import polars.selectors as cs

        from ._lazy_column_selector import _LazyColumnSelector  # circular import

        return _LazyColumnSelector(cs.all())

    @staticmethod
    def by_index(indices: int | list[int]) -> ColumnSelector:
        import polars.selectors as cs

        from ._lazy_column_selector import _LazyColumnSelector  # circular import

        return _LazyColumnSelector(cs.by_index(indices))

    @staticmethod
    def by_name(names: str | list[str], *, ignore_unknown_names: bool = False) -> ColumnSelector:
        import polars.selectors as cs

        from ._lazy_column_selector import _LazyColumnSelector  # circular import

        return _LazyColumnSelector(cs.by_name(names, require_all=not ignore_unknown_names))

    @staticmethod
    def by_type(types: ColumnType | list[ColumnType]) -> ColumnSelector:
        import polars.selectors as cs

        from ._lazy_column_selector import _LazyColumnSelector  # circular import

        return _LazyColumnSelector(cs.by_dtype(types))

    @staticmethod
    def is_float() -> ColumnSelector:
        import polars.selectors as cs

        from ._lazy_column_selector import _LazyColumnSelector  # circular import

        return _LazyColumnSelector(cs.float())

    @staticmethod
    def is_int() -> ColumnSelector:
        import polars.selectors as cs

        from ._lazy_column_selector import _LazyColumnSelector  # circular import

        return _LazyColumnSelector(cs.integer())

    @staticmethod
    def is_numeric() -> ColumnSelector:
        import polars.selectors as cs

        from ._lazy_column_selector import _LazyColumnSelector  # circular import

        return _LazyColumnSelector(cs.numeric())

    @staticmethod
    def is_signed_int() -> ColumnSelector:
        import polars.selectors as cs

        from ._lazy_column_selector import _LazyColumnSelector  # circular import

        return _LazyColumnSelector(cs.signed_integer())

    @staticmethod
    def is_temporal() -> ColumnSelector:
        import polars.selectors as cs

        from ._lazy_column_selector import _LazyColumnSelector  # circular import

        return _LazyColumnSelector(cs.temporal())

    @staticmethod
    def is_unsigned_int() -> ColumnSelector:
        import polars.selectors as cs

        from ._lazy_column_selector import _LazyColumnSelector  # circular import

        return _LazyColumnSelector(cs.unsigned_integer())

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def __invert__(self) -> ColumnSelector: ...

    @abstractmethod
    def __and__(self, other: ColumnSelector) -> ColumnSelector: ...

    @abstractmethod
    def __rand__(self, other: ColumnSelector) -> ColumnSelector: ...

    @abstractmethod
    def __or__(self, other: ColumnSelector) -> ColumnSelector: ...

    @abstractmethod
    def __ror__(self, other: ColumnSelector) -> ColumnSelector: ...

    @abstractmethod
    def __sub__(self, other: ColumnSelector) -> ColumnSelector: ...

    @abstractmethod
    def __rsub__(self, other: ColumnSelector) -> ColumnSelector: ...

    @abstractmethod
    def __xor__(self, other: ColumnSelector) -> ColumnSelector: ...

    @abstractmethod
    def __rxor__(self, other: ColumnSelector) -> ColumnSelector: ...

    # Other --------------------------------------------------------------------

    @abstractmethod
    def __eq__(self, other: object) -> bool: ...

    @abstractmethod
    def __hash__(self) -> int: ...

    @abstractmethod
    def __repr__(self) -> str: ...

    @abstractmethod
    def __sizeof__(self) -> int: ...

    @abstractmethod
    def __str__(self) -> str: ...

    # ------------------------------------------------------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------------------------------------------------------

    def not_(self) -> ColumnSelector:
        return self.__invert__()

    def and_(self, other: ColumnSelector) -> ColumnSelector:
        return self.__and__(other)

    def or_(self, other: ColumnSelector) -> ColumnSelector:
        return self.__or__(other)

    def xor(self, other: ColumnSelector) -> ColumnSelector:
        return self.__xor__(other)

    def sub(self, other: ColumnSelector) -> ColumnSelector:
        return self.__sub__(other)

    # ------------------------------------------------------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------------------------------------------------------

    @property
    @abstractmethod
    def _polars_expression(self) -> pl.Expr:
        """The polars expression that corresponds to this selector."""
