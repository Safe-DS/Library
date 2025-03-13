from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from safeds.data.tabular.typing import ColumnType

if TYPE_CHECKING:
    from polars._typing import SelectorType


class ColumnSelector:
    # ------------------------------------------------------------------------------------------------------------------
    # Static methods
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def all() -> ColumnSelector:
        """
        Select all columns.

        Returns
        -------
        selector:
            The selector.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> from safeds.data.tabular.query import ColumnSelector
        >>> table = Table({"a": [1, 2, 3], "b": ["a", "b", "c"]})
        >>> table.select_columns(ColumnSelector.all())
        +-----+-----+
        |   a | b   |
        | --- | --- |
        | i64 | str |
        +===========+
        |   1 | a   |
        |   2 | b   |
        |   3 | c   |
        +-----+-----+
        """
        import polars.selectors as cs

        from ._lazy_column_selector import _LazyColumnSelector  # circular import

        return _LazyColumnSelector(cs.all())

    @staticmethod
    def by_index(indices: int | list[int]) -> ColumnSelector:
        """
        Select columns by their index.

        Parameters
        ----------
        indices:
            The indices to select.

        Returns
        -------
        selector:
            The selector.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> from safeds.data.tabular.query import ColumnSelector
        >>> table = Table({"a": [1, 2, 3], "b": ["a", "b", "c"]})
        >>> table.select_columns(ColumnSelector.by_index(0))
        +-----+
        |   a |
        | --- |
        | i64 |
        +=====+
        |   1 |
        |   2 |
        |   3 |
        +-----+
        """
        import polars.selectors as cs

        from ._lazy_column_selector import _LazyColumnSelector  # circular import

        return _LazyColumnSelector(cs.by_index(indices))

    @staticmethod
    def by_name(names: str | list[str], *, ignore_unknown_names: bool = False) -> ColumnSelector:
        """
        Select columns by their name.

        Parameters
        ----------
        names:
            The names to select.
        ignore_unknown_names:
            If set to True, columns that are not present in the table will be ignored.
            If set to False, an error will be raised if any of the specified columns do not exist.

        Returns
        -------
        selector:
            The selector.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> from safeds.data.tabular.query import ColumnSelector
        >>> table = Table({"a": [1, 2, 3], "b": ["a", "b", "c"]})
        >>> table.select_columns(ColumnSelector.by_name("a"))
        +-----+
        |   a |
        | --- |
        | i64 |
        +=====+
        |   1 |
        |   2 |
        |   3 |
        +-----+
        """
        import polars.selectors as cs

        from ._lazy_column_selector import _LazyColumnSelector  # circular import

        return _LazyColumnSelector(cs.by_name(names, require_all=not ignore_unknown_names))

    @staticmethod
    def by_type(types: ColumnType | list[ColumnType]) -> ColumnSelector:
        """
        Select all columns that have a specific type.

        Parameters
        ----------
        types:
            The column types to select.

        Returns
        -------
        selector:
            The selector.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> from safeds.data.tabular.query import ColumnSelector
        >>> from safeds.data.tabular.typing import ColumnType
        >>> table = Table({"a": [1, 2, 3], "b": ["a", "b", "c"]})
        >>> table.select_columns(ColumnSelector.by_type(ColumnType.string()))
        +-----+
        | b   |
        | --- |
        | str |
        +=====+
        | a   |
        | b   |
        | c   |
        +-----+
        """
        import polars.selectors as cs

        from ._lazy_column_selector import _LazyColumnSelector  # circular import

        if isinstance(types, ColumnType):
            types = [types]

        types = [t._polars_data_type for t in types]

        return _LazyColumnSelector(cs.by_dtype(types))

    @staticmethod
    def is_float() -> ColumnSelector:
        """
        Select all columns that have a floating point type.

        Returns
        -------
        selector:
            The selector.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> from safeds.data.tabular.query import ColumnSelector
        >>> table = Table({"a": [1.0, 2.0, 3.0], "b": ["a", "b", "c"]})
        >>> table.select_columns(ColumnSelector.is_float())
        +---------+
        |       a |
        |     --- |
        |     f64 |
        +=========+
        | 1.00000 |
        | 2.00000 |
        | 3.00000 |
        +---------+
        """
        import polars.selectors as cs

        from ._lazy_column_selector import _LazyColumnSelector  # circular import

        return _LazyColumnSelector(cs.float())

    @staticmethod
    def is_int() -> ColumnSelector:
        """
        Select all columns that have an integer type.

        Returns
        -------
        selector:
            The selector.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> from safeds.data.tabular.query import ColumnSelector
        >>> table = Table({"a": [1, 2, 3], "b": ["a", "b", "c"]})
        >>> table.select_columns(ColumnSelector.is_int())
        +-----+
        |   a |
        | --- |
        | i64 |
        +=====+
        |   1 |
        |   2 |
        |   3 |
        +-----+
        """
        import polars.selectors as cs

        from ._lazy_column_selector import _LazyColumnSelector  # circular import

        return _LazyColumnSelector(cs.integer())

    @staticmethod
    def is_numeric() -> ColumnSelector:
        """
        Select all columns that have a numeric type.

        Returns
        -------
        selector:
            The selector.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> from safeds.data.tabular.query import ColumnSelector
        >>> table = Table({"a": [1, 2, 3], "b": ["a", "b", "c"]})
        >>> table.select_columns(ColumnSelector.is_numeric())
        +-----+
        |   a |
        | --- |
        | i64 |
        +=====+
        |   1 |
        |   2 |
        |   3 |
        +-----+
        """
        import polars.selectors as cs

        from ._lazy_column_selector import _LazyColumnSelector  # circular import

        return _LazyColumnSelector(cs.numeric())

    @staticmethod
    def is_signed_int() -> ColumnSelector:
        """
        Select all columns that have a signed integer type.

        Returns
        -------
        selector:
            The selector.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> from safeds.data.tabular.query import ColumnSelector
        >>> table = Table({"a": [1, 2, 3], "b": ["a", "b", "c"]})
        >>> table.select_columns(ColumnSelector.is_signed_int())
        +-----+
        |   a |
        | --- |
        | i64 |
        +=====+
        |   1 |
        |   2 |
        |   3 |
        +-----+
        """
        import polars.selectors as cs

        from ._lazy_column_selector import _LazyColumnSelector  # circular import

        return _LazyColumnSelector(cs.signed_integer())

    @staticmethod
    def is_temporal() -> ColumnSelector:
        """
        Select all columns that have a temporal type.

        Returns
        -------
        selector:
            The selector.

        Examples
        --------
        >>> from datetime import date
        >>> from safeds.data.tabular.containers import Table
        >>> from safeds.data.tabular.query import ColumnSelector
        >>> table = Table({
        ...     "a": [date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3)],
        ...     "b": ["a", "b", "c"]
        ... })
        >>> table.select_columns(ColumnSelector.is_temporal())
        +------------+
        | a          |
        | ---        |
        | date       |
        +============+
        | 2022-01-01 |
        | 2022-01-02 |
        | 2022-01-03 |
        +------------+
        """
        import polars.selectors as cs

        from ._lazy_column_selector import _LazyColumnSelector  # circular import

        return _LazyColumnSelector(cs.temporal())

    @staticmethod
    def is_unsigned_int() -> ColumnSelector:
        """
        Select all columns that have a unsigned integer type.

        Returns
        -------
        selector:
            The selector.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column, Table
        >>> from safeds.data.tabular.query import ColumnSelector
        >>> table = Table({"b": ["a", "b", "c"]})
        >>> column = Column("a", [1, 2, 3], type=ColumnType.uint8())
        >>> table = table.add_columns(column)
        >>> table.select_columns(ColumnSelector.is_unsigned_int())
        +-----+
        |   a |
        | --- |
        |  u8 |
        +=====+
        |   1 |
        |   2 |
        |   3 |
        +-----+
        """
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
        """
        Invert the selector. This is equivalent to the `~` operator.

        Do **not** use the `not` operator. Its behavior cannot be overwritten in Python, so it will not work as
        expected.

        Returns
        -------
        selector:
            The inverted selector.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> from safeds.data.tabular.query import ColumnSelector
        >>> table = Table({"a": [1, 2, 3], "b": ["a", "b", "c"]})
        >>> table.select_columns(ColumnSelector.is_numeric().not_())
        +-----+
        | b   |
        | --- |
        | str |
        +=====+
        | a   |
        | b   |
        | c   |
        +-----+

        >>> table.select_columns(~ColumnSelector.is_numeric())
        +-----+
        | b   |
        | --- |
        | str |
        +=====+
        | a   |
        | b   |
        | c   |
        +-----+
        """
        return self.__invert__()

    def and_(self, other: ColumnSelector) -> ColumnSelector:
        """
        Create the intersection of two selectors. This is equivalent to the `&` operator.

        Parameters
        ----------
        other:
            The other selector.

        Returns
        -------
        selector:
            The intersection of the two selectors.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> from safeds.data.tabular.query import ColumnSelector
        >>> table = Table({"a": [1, 2, 3], "b": ["a", "b", "c"]})
        >>> table.select_columns(
        ...     ColumnSelector.is_numeric().and_(ColumnSelector.by_name("a"))
        ... )
        +-----+
        |   a |
        | --- |
        | i64 |
        +=====+
        |   1 |
        |   2 |
        |   3 |
        +-----+

        >>> table.select_columns(
        ...     ColumnSelector.is_numeric() & ColumnSelector.by_name("a")
        ... )
        +-----+
        |   a |
        | --- |
        | i64 |
        +=====+
        |   1 |
        |   2 |
        |   3 |
        +-----+
        """
        return self.__and__(other)

    def or_(self, other: ColumnSelector) -> ColumnSelector:
        """
        Create the union of two selectors. This is equivalent to the `|` operator.

        Parameters
        ----------
        other:
            The other selector.

        Returns
        -------
        selector:
            The union of the two selectors.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> from safeds.data.tabular.query import ColumnSelector
        >>> table = Table({"a": [1, 2, 3], "b": ["a", "b", "c"]})
        >>> table.select_columns(
        ...     ColumnSelector.is_numeric().or_(ColumnSelector.by_name("b"))
        ... )
        +-----+-----+
        |   a | b   |
        | --- | --- |
        | i64 | str |
        +===========+
        |   1 | a   |
        |   2 | b   |
        |   3 | c   |
        +-----+-----+

        >>> table.select_columns(
        ...     ColumnSelector.is_numeric() | ColumnSelector.by_name("b")
        ... )
        +-----+-----+
        |   a | b   |
        | --- | --- |
        | i64 | str |
        +===========+
        |   1 | a   |
        |   2 | b   |
        |   3 | c   |
        +-----+-----+
        """
        return self.__or__(other)

    def xor(self, other: ColumnSelector) -> ColumnSelector:
        """
        Create the symmetric difference of two selectors. This is equivalent to the `^` operator.

        Parameters
        ----------
        other:
            The other selector.

        Returns
        -------
        selector:
            The symmetric difference of the two selectors.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> from safeds.data.tabular.query import ColumnSelector
        >>> table = Table({"a": [1, 2, 3], "b": ["a", "b", "c"]})
        >>> table.select_columns(
        ...     ColumnSelector.is_numeric().xor(ColumnSelector.by_name("b"))
        ... )
        +-----+-----+
        |   a | b   |
        | --- | --- |
        | i64 | str |
        +===========+
        |   1 | a   |
        |   2 | b   |
        |   3 | c   |
        +-----+-----+

        >>> table.select_columns(
        ...     ColumnSelector.is_numeric() ^ ColumnSelector.by_name("b")
        ... )
        +-----+-----+
        |   a | b   |
        | --- | --- |
        | i64 | str |
        +===========+
        |   1 | a   |
        |   2 | b   |
        |   3 | c   |
        +-----+-----+
        """
        return self.__xor__(other)

    def sub(self, other: ColumnSelector) -> ColumnSelector:
        """
        Create the difference of two selectors. This is equivalent to the `-` operator.

        Parameters
        ----------
        other:
            The other selector.

        Returns
        -------
        selector:
            The difference of the two selectors.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table
        >>> from safeds.data.tabular.query import ColumnSelector
        >>> table = Table({"a": [1, 2, 3], "b": ["a", "b", "c"]})
        >>> table.select_columns(
        ...     ColumnSelector.is_numeric().sub(ColumnSelector.by_name("b"))
        ... )
        +-----+
        |   a |
        | --- |
        | i64 |
        +=====+
        |   1 |
        |   2 |
        |   3 |
        +-----+

        >>> table.select_columns(
        ...     ColumnSelector.is_numeric() - ColumnSelector.by_name("b")
        ... )
        +-----+
        |   a |
        | --- |
        | i64 |
        +=====+
        |   1 |
        |   2 |
        |   3 |
        +-----+
        """
        return self.__sub__(other)

    # ------------------------------------------------------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------------------------------------------------------

    @property
    @abstractmethod
    def _polars_selector(self) -> SelectorType:
        """The polars expression that corresponds to this selector."""
