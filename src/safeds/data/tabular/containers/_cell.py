from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    import datetime as python_datetime

    import polars as pl

    from safeds._typing import (
        _BooleanCell,
        _ConvertibleToBooleanCell,
        _ConvertibleToCell,
        _ConvertibleToIntCell,
        _PythonLiteral,
    )
    from safeds.data.tabular.typing import ColumnType

    from ._string_cell import StringCell
    from ._temporal_cell import TemporalCell

T_co = TypeVar("T_co", covariant=True)
P = TypeVar("P")


class Cell(ABC, Generic[T_co]):
    """
    A single value in a table.

    You only need to interact with this class in callbacks passed to higher-order functions.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Static methods
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def constant(value: _PythonLiteral | None) -> Cell:
        """
        Create a cell with a constant value.

        Parameters
        ----------
        value:
            The value to create the cell from.

        Returns
        -------
        cell:
            The created cell.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, None])
        >>> column.transform(lambda _: Cell.constant(3))
        +-----+
        |   a |
        | --- |
        | i32 |
        +=====+
        |   3 |
        |   3 |
        |   3 |
        +-----+
        """
        import polars as pl

        from ._lazy_cell import _LazyCell  # circular import

        return _LazyCell(pl.lit(value))

    @staticmethod
    def date(
        year: _ConvertibleToIntCell,
        month: _ConvertibleToIntCell,
        day: _ConvertibleToIntCell,
    ) -> Cell[python_datetime.date | None]:
        """
        Create a cell with a date.

        Invalid dates are converted to missing values (`None`).

        Parameters
        ----------
        year:
            The year.
        month:
            The month. Must be between 1 and 12.
        day:
            The day. Must be between 1 and 31.

        Returns
        -------
        cell:
            The created cell.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, None])
        >>> column.transform(lambda _: Cell.date(2025, 1, 15))
        +------------+
        | a          |
        | ---        |
        | date       |
        +============+
        | 2025-01-15 |
        | 2025-01-15 |
        | 2025-01-15 |
        +------------+

        >>> column.transform(lambda cell: Cell.date(2025, cell, 15))
        +------------+
        | a          |
        | ---        |
        | date       |
        +============+
        | 2025-01-15 |
        | 2025-02-15 |
        | null       |
        +------------+
        """
        import polars as pl

        from ._lazy_cell import _LazyCell  # circular import

        year = _unwrap(year)
        month = _unwrap(month)
        day = _unwrap(day)

        return _LazyCell(pl.date(year, month, day))

    @staticmethod
    def datetime(
        year: _ConvertibleToIntCell,
        month: _ConvertibleToIntCell,
        day: _ConvertibleToIntCell,
        *,
        hour: _ConvertibleToIntCell = 0,
        minute: _ConvertibleToIntCell = 0,
        second: _ConvertibleToIntCell = 0,
        microsecond: _ConvertibleToIntCell = 0,
    ) -> Cell[python_datetime.datetime | None]:
        """
        Create a cell with a datetime.

        Invalid datetimes are converted to missing values (`None`).

        Parameters
        ----------
        year:
            The year.
        month:
            The month. Must be between 1 and 12.
        day:
            The day. Must be between 1 and 31.
        hour:
            The hour. Must be between 0 and 23.
        minute:
            The minute. Must be between 0 and 59.
        second:
            The second. Must be between 0 and 59.
        microsecond:
            The microsecond. Must be between 0 and 999,999.

        Returns
        -------
        cell:
            The created cell.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, None])
        >>> column.transform(lambda _: Cell.datetime(2025, 1, 15, hour=12))
        +---------------------+
        | a                   |
        | ---                 |
        | datetime[μs]        |
        +=====================+
        | 2025-01-15 12:00:00 |
        | 2025-01-15 12:00:00 |
        | 2025-01-15 12:00:00 |
        +---------------------+

        >>> column.transform(lambda cell: Cell.datetime(2025, 1, 15, hour=cell))
        +---------------------+
        | a                   |
        | ---                 |
        | datetime[μs]        |
        +=====================+
        | 2025-01-15 01:00:00 |
        | 2025-01-15 02:00:00 |
        | null                |
        +---------------------+
        """
        import polars as pl

        from ._lazy_cell import _LazyCell  # circular import

        year = _unwrap(year)
        month = _unwrap(month)
        day = _unwrap(day)
        hour = _unwrap(hour)
        minute = _unwrap(minute)
        second = _unwrap(second)
        microsecond = _unwrap(microsecond)

        # By default, microseconds overflow into seconds
        return _LazyCell(
            pl.when(microsecond <= 999_999)
            .then(pl.datetime(year, month, day, hour, minute, second, microsecond))
            .otherwise(None),
        )

    @staticmethod
    def duration(
        *,
        weeks: _ConvertibleToIntCell = 0,
        days: _ConvertibleToIntCell = 0,
        hours: _ConvertibleToIntCell = 0,
        minutes: _ConvertibleToIntCell = 0,
        seconds: _ConvertibleToIntCell = 0,
        milliseconds: _ConvertibleToIntCell = 0,
        microseconds: _ConvertibleToIntCell = 0,
    ) -> Cell[python_datetime.timedelta | None]:
        """
        Create a cell with a duration.

        Invalid durations are converted to missing values (`None`).

        Parameters
        ----------
        weeks:
            The number of weeks.
        days:
            The number of days.
        hours:
            The number of hours.
        minutes:
            The number of minutes.
        seconds:
            The number of seconds.
        milliseconds:
            The number of milliseconds.
        microseconds:
            The number of microseconds.

        Returns
        -------
        cell:
            The created cell.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, None])
        >>> column.transform(lambda _: Cell.duration(hours=1))
        +--------------+
        | a            |
        | ---          |
        | duration[μs] |
        +==============+
        | 1h           |
        | 1h           |
        | 1h           |
        +--------------+

        >>> column.transform(lambda cell: Cell.duration(hours = cell))
        +--------------+
        | a            |
        | ---          |
        | duration[μs] |
        +==============+
        | 1h           |
        | 2h           |
        | null         |
        +--------------+
        """
        import polars as pl

        from ._lazy_cell import _LazyCell  # circular import

        weeks = _unwrap(weeks)
        days = _unwrap(days)
        hours = _unwrap(hours)
        minutes = _unwrap(minutes)
        seconds = _unwrap(seconds)
        milliseconds = _unwrap(milliseconds)
        microseconds = _unwrap(microseconds)

        return _LazyCell(
            pl.duration(
                weeks=weeks,
                days=days,
                hours=hours,
                minutes=minutes,
                seconds=seconds,
                milliseconds=milliseconds,
                microseconds=microseconds,
            ),
        )

    @staticmethod
    def time(
        hour: int | Cell[int],
        minute: int | Cell[int],
        second: int | Cell[int],
        *,
        microsecond: int | Cell[int] = 0,
    ) -> Cell[python_datetime.time | None]:
        """
        Create a cell with a time.

        Invalid times are converted to missing values (`None`).

        Parameters
        ----------
        hour:
            The hour. Must be between 0 and 23.
        minute:
            The minute. Must be between 0 and 59.
        second:
            The second. Must be between 0 and 59.
        microsecond:
            The microsecond. Must be between 0 and 999,999.

        Returns
        -------
        cell:
            The created cell.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, None])
        >>> column.transform(lambda _: Cell.time(12, 0, 0))
        +----------+
        | a        |
        | ---      |
        | time     |
        +==========+
        | 12:00:00 |
        | 12:00:00 |
        | 12:00:00 |
        +----------+

        >>> column.transform(lambda cell: Cell.time(12, cell, 0, microsecond=1))
        +-----------------+
        | a               |
        | ---             |
        | time            |
        +=================+
        | 12:01:00.000001 |
        | 12:02:00.000001 |
        | null            |
        +-----------------+
        """
        import polars as pl

        from ._lazy_cell import _LazyCell  # circular import

        hour = _unwrap(hour)
        minute = _unwrap(minute)
        second = _unwrap(second)
        microsecond = _unwrap(microsecond)

        # By default, microseconds overflow into seconds
        return _LazyCell(
            pl.when(microsecond <= 999_999).then(pl.time(hour, minute, second, microsecond)).otherwise(None),
        )

    @staticmethod
    def first_not_none(cells: list[Cell[P]]) -> Cell[P | None]:
        """
        Return the first cell that is not None or None if all cells are None.

        Parameters
        ----------
        cells:
            The list of cells to be checked.

        Returns
        -------
        cell:
            The first cell that is not None or None if all cells are None.
        """
        import polars as pl

        from ._lazy_cell import _LazyCell  # circular import

        # `coalesce` raises in this case
        if not cells:
            return Cell.constant(None)

        return _LazyCell(pl.coalesce([_unwrap(cell) for cell in cells]))

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    # "Boolean" operators (actually bitwise) -----------------------------------

    @abstractmethod
    def __invert__(self) -> _BooleanCell: ...

    @abstractmethod
    def __and__(self, other: _ConvertibleToBooleanCell) -> _BooleanCell: ...

    @abstractmethod
    def __rand__(self, other: _ConvertibleToBooleanCell) -> _BooleanCell: ...

    @abstractmethod
    def __or__(self, other: _ConvertibleToBooleanCell) -> _BooleanCell: ...

    @abstractmethod
    def __ror__(self, other: _ConvertibleToBooleanCell) -> _BooleanCell: ...

    @abstractmethod
    def __xor__(self, other: _ConvertibleToBooleanCell) -> _BooleanCell: ...

    @abstractmethod
    def __rxor__(self, other: _ConvertibleToBooleanCell) -> _BooleanCell: ...

    # Comparison ---------------------------------------------------------------

    @abstractmethod
    def __eq__(self, other: _ConvertibleToCell) -> _BooleanCell:  # type: ignore[override]
        ...

    @abstractmethod
    def __ge__(self, other: _ConvertibleToCell) -> _BooleanCell: ...

    @abstractmethod
    def __gt__(self, other: _ConvertibleToCell) -> _BooleanCell: ...

    @abstractmethod
    def __le__(self, other: _ConvertibleToCell) -> _BooleanCell: ...

    @abstractmethod
    def __lt__(self, other: _ConvertibleToCell) -> _BooleanCell: ...

    @abstractmethod
    def __ne__(self, other: _ConvertibleToCell) -> _BooleanCell:  # type: ignore[override]
        ...

    # Numeric operators --------------------------------------------------------

    @abstractmethod
    def __abs__(self) -> Cell: ...

    @abstractmethod
    def __ceil__(self) -> Cell: ...

    @abstractmethod
    def __floor__(self) -> Cell: ...

    @abstractmethod
    def __neg__(self) -> Cell: ...

    @abstractmethod
    def __pos__(self) -> Cell: ...

    @abstractmethod
    def __add__(self, other: _ConvertibleToCell) -> Cell: ...

    @abstractmethod
    def __radd__(self, other: _ConvertibleToCell) -> Cell: ...

    @abstractmethod
    def __floordiv__(self, other: _ConvertibleToCell) -> Cell: ...

    @abstractmethod
    def __rfloordiv__(self, other: _ConvertibleToCell) -> Cell: ...

    @abstractmethod
    def __mod__(self, other: _ConvertibleToCell) -> Cell: ...

    @abstractmethod
    def __rmod__(self, other: _ConvertibleToCell) -> Cell: ...

    @abstractmethod
    def __mul__(self, other: _ConvertibleToCell) -> Cell: ...

    @abstractmethod
    def __rmul__(self, other: _ConvertibleToCell) -> Cell: ...

    @abstractmethod
    def __pow__(self, other: _ConvertibleToCell) -> Cell: ...

    @abstractmethod
    def __rpow__(self, other: _ConvertibleToCell) -> Cell: ...

    @abstractmethod
    def __sub__(self, other: _ConvertibleToCell) -> Cell: ...

    @abstractmethod
    def __rsub__(self, other: _ConvertibleToCell) -> Cell: ...

    @abstractmethod
    def __truediv__(self, other: _ConvertibleToCell) -> Cell: ...

    @abstractmethod
    def __rtruediv__(self, other: _ConvertibleToCell) -> Cell: ...

    # Other --------------------------------------------------------------------

    @abstractmethod
    def __hash__(self) -> int: ...

    @abstractmethod
    def __repr__(self) -> str: ...

    @abstractmethod
    def __sizeof__(self) -> int: ...

    @abstractmethod
    def __str__(self) -> str: ...

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    @abstractmethod
    def str(self) -> StringCell:
        """
        Namespace for operations on strings.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", ["hi", "hello"])
        >>> column.transform(lambda cell: cell.str.length())
        +-----+
        |   a |
        | --- |
        | u32 |
        +=====+
        |   2 |
        |   5 |
        +-----+
        """

    @property
    @abstractmethod
    def dt(self) -> TemporalCell:
        """
        Namespace for operations on temporal values.

        Examples
        --------
        >>> import datetime
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [datetime.datetime(2025, 1, 1), datetime.datetime(2024, 1, 1)])
        >>> column.transform(lambda cell: cell.dt.year())
        +------+
        |    a |
        |  --- |
        |  i32 |
        +======+
        | 2025 |
        | 2024 |
        +------+
        """

    # ------------------------------------------------------------------------------------------------------------------
    # Boolean operations
    # ------------------------------------------------------------------------------------------------------------------

    def not_(self) -> _BooleanCell:
        """
        Negate a boolean. This is equivalent to the `~` operator.

        Do **not** use the `not` operator. Its behavior cannot be overwritten in Python, so it will not work as
        expected.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [True, False, None])
        >>> column.transform(lambda cell: cell.not_())
        +-------+
        | a     |
        | ---   |
        | bool  |
        +=======+
        | false |
        | true  |
        | null  |
        +-------+

        >>> column.transform(lambda cell: ~cell)
        +-------+
        | a     |
        | ---   |
        | bool  |
        +=======+
        | false |
        | true  |
        | null  |
        +-------+
        """
        return self.__invert__()

    def and_(self, other: _ConvertibleToBooleanCell) -> _BooleanCell:
        """
        Perform a boolean AND operation. This is equivalent to the `&` operator.

        Do **not** use the `and` operator. Its behavior cannot be overwritten in Python, so it will not work as
        expected.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [True, False, None])
        >>> column.transform(lambda cell: cell.and_(True))
        +-------+
        | a     |
        | ---   |
        | bool  |
        +=======+
        | true  |
        | false |
        | null  |
        +-------+

        >>> column.transform(lambda cell: cell & True)
        +-------+
        | a     |
        | ---   |
        | bool  |
        +=======+
        | true  |
        | false |
        | null  |
        +-------+
        """
        return self.__and__(other)

    def or_(self, other: _ConvertibleToBooleanCell) -> _BooleanCell:
        """
        Perform a boolean OR operation. This is equivalent to the `|` operator.

        Do **not** use the `or` operator. Its behavior cannot be overwritten in Python, so it will not work as expected.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [True, False, None])
        >>> column.transform(lambda cell: cell.or_(False))
        +-------+
        | a     |
        | ---   |
        | bool  |
        +=======+
        | true  |
        | false |
        | null  |
        +-------+

        >>> column.transform(lambda cell: cell | False)
        +-------+
        | a     |
        | ---   |
        | bool  |
        +=======+
        | true  |
        | false |
        | null  |
        +-------+
        """
        return self.__or__(other)

    def xor(self, other: _ConvertibleToBooleanCell) -> _BooleanCell:
        """
        Perform a boolean XOR operation. This is equivalent to the `^` operator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [True, False, None])
        >>> column.transform(lambda cell: cell.xor(True))
        +-------+
        | a     |
        | ---   |
        | bool  |
        +=======+
        | false |
        | true  |
        | null  |
        +-------+

        >>> column.transform(lambda cell: cell ^ True)
        +-------+
        | a     |
        | ---   |
        | bool  |
        +=======+
        | false |
        | true  |
        | null  |
        +-------+
        """
        return self.__xor__(other)

    # ------------------------------------------------------------------------------------------------------------------
    # Numeric operations
    # ------------------------------------------------------------------------------------------------------------------

    def abs(self) -> Cell:
        """
        Get the absolute value.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, -2, None])
        >>> column.transform(lambda cell: cell.abs())
        +------+
        |    a |
        |  --- |
        |  i64 |
        +======+
        |    1 |
        |    2 |
        | null |
        +------+
        """
        return self.__abs__()

    def ceil(self) -> Cell:
        """
        Round up to the nearest integer.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1.1, 3.0, None])
        >>> column.transform(lambda cell: cell.ceil())
        +---------+
        |       a |
        |     --- |
        |     f64 |
        +=========+
        | 2.00000 |
        | 3.00000 |
        |    null |
        +---------+
        """
        return self.__ceil__()

    def floor(self) -> Cell:
        """
        Round down to the nearest integer.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1.1, 3.0, None])
        >>> column.transform(lambda cell: cell.floor())
        +---------+
        |       a |
        |     --- |
        |     f64 |
        +=========+
        | 1.00000 |
        | 3.00000 |
        |    null |
        +---------+
        """
        return self.__floor__()

    def neg(self) -> Cell:
        """
        Negate the value. This is equivalent to the unary `-` operator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, -2, None])
        >>> column.transform(lambda cell: cell.neg())
        +------+
        |    a |
        |  --- |
        |  i64 |
        +======+
        |   -1 |
        |    2 |
        | null |
        +------+

        >>> column.transform(lambda cell: -cell)
        +------+
        |    a |
        |  --- |
        |  i64 |
        +======+
        |   -1 |
        |    2 |
        | null |
        +------+
        """
        return self.__neg__()

    def add(self, other: _ConvertibleToCell) -> Cell:
        """
        Add a value. This is equivalent to the `+` operator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, None])
        >>> column.transform(lambda cell: cell.add(3))
        +------+
        |    a |
        |  --- |
        |  i64 |
        +======+
        |    4 |
        |    5 |
        | null |
        +------+

        >>> column.transform(lambda cell: cell + 3)
        +------+
        |    a |
        |  --- |
        |  i64 |
        +======+
        |    4 |
        |    5 |
        | null |
        +------+
        """
        return self.__add__(other)

    def div(self, other: _ConvertibleToCell) -> Cell:
        """
        Divide by a value. This is equivalent to the `/` operator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [6, 8, None])
        >>> column.transform(lambda cell: cell.div(2))
        +---------+
        |       a |
        |     --- |
        |     f64 |
        +=========+
        | 3.00000 |
        | 4.00000 |
        |    null |
        +---------+

        >>> column.transform(lambda cell: cell / 2)
        +---------+
        |       a |
        |     --- |
        |     f64 |
        +=========+
        | 3.00000 |
        | 4.00000 |
        |    null |
        +---------+
        """
        return self.__truediv__(other)

    def mod(self, other: _ConvertibleToCell) -> Cell:
        """
        Perform a modulo operation. This is equivalent to the `%` operator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [5, 6, -1, None])
        >>> column.transform(lambda cell: cell.mod(3))
        +------+
        |    a |
        |  --- |
        |  i64 |
        +======+
        |    2 |
        |    0 |
        |    2 |
        | null |
        +------+

        >>> column.transform(lambda cell: cell % 3)
        +------+
        |    a |
        |  --- |
        |  i64 |
        +======+
        |    2 |
        |    0 |
        |    2 |
        | null |
        +------+
        """
        return self.__mod__(other)

    def mul(self, other: _ConvertibleToCell) -> Cell:
        """
        Multiply by a value. This is equivalent to the `*` operator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [2, 3, None])
        >>> column.transform(lambda cell: cell.mul(4))
        +------+
        |    a |
        |  --- |
        |  i64 |
        +======+
        |    8 |
        |   12 |
        | null |
        +------+

        >>> column.transform(lambda cell: cell * 4)
        +------+
        |    a |
        |  --- |
        |  i64 |
        +======+
        |    8 |
        |   12 |
        | null |
        +------+
        """
        return self.__mul__(other)

    def pow(self, other: _ConvertibleToCell) -> Cell:
        """
        Raise to a power. This is equivalent to the `**` operator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [2, 3, None])
        >>> column.transform(lambda cell: cell.pow(3))
        +------+
        |    a |
        |  --- |
        |  i64 |
        +======+
        |    8 |
        |   27 |
        | null |
        +------+


        >>> column.transform(lambda cell: cell ** 3)
        +------+
        |    a |
        |  --- |
        |  i64 |
        +======+
        |    8 |
        |   27 |
        | null |
        +------+
        """
        return self.__pow__(other)

    def sub(self, other: _ConvertibleToCell) -> Cell:
        """
        Subtract a value. This is equivalent to the binary `-` operator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [5, 6, None])
        >>> column.transform(lambda cell: cell.sub(3))
        +------+
        |    a |
        |  --- |
        |  i64 |
        +======+
        |    2 |
        |    3 |
        | null |
        +------+

        >>> column.transform(lambda cell: cell - 3)
        +------+
        |    a |
        |  --- |
        |  i64 |
        +======+
        |    2 |
        |    3 |
        | null |
        +------+
        """
        return self.__sub__(other)

    # ------------------------------------------------------------------------------------------------------------------
    # Comparison operations
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def eq(
        self,
        other: _ConvertibleToCell,
        *,
        propagate_missing_values: bool = True,
    ) -> _BooleanCell:
        """
        Check if equal to a value. The default behavior is equivalent to the `==` operator.

        Missing values (indicated by `None`) are handled as follows:

        - If `propagate_missing_values` is `True` (default), the result will be a missing value if either the cell or
          the other value is a missing value. Here, `None == None` is `None`. The intuition is that we do not know the
          result of the comparison if we do not know the values, which is consistent with the other cell operations.
        - If `propagate_missing_values` is `False`, `None` will be treated as a regular value. Here, `None == None`
          is `True`. This behavior is useful, if you want to work with missing values, e.g. to filter them out.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, None])
        >>> column.transform(lambda cell: cell.eq(2))
        +-------+
        | a     |
        | ---   |
        | bool  |
        +=======+
        | false |
        | true  |
        | null  |
        +-------+

        >>> column.transform(lambda cell: cell == 2)
        +-------+
        | a     |
        | ---   |
        | bool  |
        +=======+
        | false |
        | true  |
        | null  |
        +-------+

        >>> column.transform(lambda cell: cell.eq(2, propagate_missing_values=False))
        +-------+
        | a     |
        | ---   |
        | bool  |
        +=======+
        | false |
        | true  |
        | false |
        +-------+
        """

    @abstractmethod
    def neq(
        self,
        other: _ConvertibleToCell,
        *,
        propagate_missing_values: bool = True,
    ) -> _BooleanCell:
        """
        Check if not equal to a value. The default behavior is equivalent to the `!=` operator.

        Missing values (indicated by `None`) are handled as follows:

        - If `propagate_missing_values` is `True` (default), the result will be a missing value if either the cell or
          the other value is a missing value. Here, `None != None` is `None`. The intuition is that we do not know the
          result of the comparison if we do not know the values, which is consistent with the other cell operations.
        - If `propagate_missing_values` is `False`, `None` will be treated as a regular value. Here, `None != None`
          is `False`. This behavior is useful, if you want to work with missing values, e.g. to filter them out.

        Parameters
        ----------
        other:
            The value to compare to.
        propagate_missing_values:
            Whether to propagate missing values.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, None])
        >>> column.transform(lambda cell: cell.neq(2))
        +-------+
        | a     |
        | ---   |
        | bool  |
        +=======+
        | true  |
        | false |
        | null  |
        +-------+

        >>> column.transform(lambda cell: cell != 2)
        +-------+
        | a     |
        | ---   |
        | bool  |
        +=======+
        | true  |
        | false |
        | null  |
        +-------+

        >>> column.transform(lambda cell: cell.neq(2, propagate_missing_values=False))
        +-------+
        | a     |
        | ---   |
        | bool  |
        +=======+
        | true  |
        | false |
        | true  |
        +-------+
        """

    def ge(self, other: _ConvertibleToCell) -> _BooleanCell:
        """
        Check if greater than or equal to a value. This is equivalent to the `>=` operator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, None])
        >>> column.transform(lambda cell: cell.ge(2))
        +-------+
        | a     |
        | ---   |
        | bool  |
        +=======+
        | false |
        | true  |
        | null  |
        +-------+

        >>> column.transform(lambda cell: cell >= 2)
        +-------+
        | a     |
        | ---   |
        | bool  |
        +=======+
        | false |
        | true  |
        | null  |
        +-------+
        """
        return self.__ge__(other)

    def gt(self, other: _ConvertibleToCell) -> _BooleanCell:
        """
        Check if greater than a value. This is equivalent to the `>` operator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, None])
        >>> column.transform(lambda cell: cell.gt(2))
        +-------+
        | a     |
        | ---   |
        | bool  |
        +=======+
        | false |
        | false |
        | null  |
        +-------+

        >>> column.transform(lambda cell: cell > 2)
        +-------+
        | a     |
        | ---   |
        | bool  |
        +=======+
        | false |
        | false |
        | null  |
        +-------+
        """
        return self.__gt__(other)

    def le(self, other: _ConvertibleToCell) -> _BooleanCell:
        """
        Check if less than or equal to a value. This is equivalent to the `<=` operator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, None])
        >>> column.transform(lambda cell: cell.le(2))
        +------+
        | a    |
        | ---  |
        | bool |
        +======+
        | true |
        | true |
        | null |
        +------+

        >>> column.transform(lambda cell: cell <= 2)
        +------+
        | a    |
        | ---  |
        | bool |
        +======+
        | true |
        | true |
        | null |
        +------+
        """
        return self.__le__(other)

    def lt(self, other: _ConvertibleToCell) -> _BooleanCell:
        """
        Check if less than a value. This is equivalent to the `<` operator.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [1, 2, None])
        >>> column.transform(lambda cell: cell.lt(2))
        +-------+
        | a     |
        | ---   |
        | bool  |
        +=======+
        | true  |
        | false |
        | null  |
        +-------+

        >>> column.transform(lambda cell: cell < 2)
        +-------+
        | a     |
        | ---   |
        | bool  |
        +=======+
        | true  |
        | false |
        | null  |
        +-------+
        """
        return self.__lt__(other)

    # ------------------------------------------------------------------------------------------------------------------
    # Other
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def cast(self, type_: ColumnType) -> Cell:
        """
        Cast the cell to a different type.

        Parameters
        ----------
        type_:
            The type to cast to.

        Returns
        -------
        cell:
            The cast cell.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> from safeds.data.tabular.typing import ColumnType
        >>> column = Column("a", [1, 2, None])
        >>> column.transform(lambda cell: cell.cast(ColumnType.string()))
        +------+
        | a    |
        | ---  |
        | str  |
        +======+
        | 1    |
        | 2    |
        | null |
        +------+
        """

    # ------------------------------------------------------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------------------------------------------------------

    @property
    @abstractmethod
    def _polars_expression(self) -> pl.Expr:
        """The Polars expression that corresponds to this cell."""

    @abstractmethod
    def _equals(self, other: object) -> bool:
        """
        Check if this cell is equal to another object.

        This method is needed because the `__eq__` method is used for element-wise comparisons.
        """


def _unwrap(cell_proxy: _ConvertibleToCell) -> pl.Expr:
    import polars as pl

    if isinstance(cell_proxy, Cell):
        return cell_proxy._polars_expression
    else:
        return pl.lit(cell_proxy)
