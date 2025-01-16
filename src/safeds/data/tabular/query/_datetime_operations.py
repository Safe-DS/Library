from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import datetime as python_datetime

    from safeds._typing import _ConvertibleToIntCell
    from safeds.data.tabular.containers import Cell


class DatetimeOperations(ABC):
    """
    Namespace for operations on datetimes, dates, and times.

    This class cannot be instantiated directly. It can only be accessed using the `dt` attribute of a cell.

    Examples
    --------
    >>> from datetime import date
    >>> from safeds.data.tabular.containers import Column
    >>> column = Column("a", [date(2022, 1, 9), date(2024, 6, 12)])
    >>> column.transform(lambda cell: cell.dt.year())
    +------+
    |    a |
    |  --- |
    |  i32 |
    +======+
    | 2022 |
    | 2024 |
    +------+
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

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
    # Extract components
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def century(self) -> Cell[int | None]:
        """
        Extract the century from a datetime or date.

        Note that since our calendar begins with year 1 the first century lasts from year 1 to year 100. Subsequent
        centuries begin with years ending in "01" and end with years ending in "00".

        Returns
        -------
        cell:
            The century.

        Examples
        --------
        >>> from datetime import datetime, date
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("a", [datetime(1999, 12, 31), datetime(2000, 1, 1), datetime(2001, 1, 1), None])
        >>> column1.transform(lambda cell: cell.dt.century())
        +------+
        |    a |
        |  --- |
        |  i32 |
        +======+
        |   20 |
        |   20 |
        |   21 |
        | null |
        +------+

        >>> column2 = Column("a", [date(1999, 12, 31), date(2000, 1, 1), date(2001, 1, 1), None])
        >>> column2.transform(lambda cell: cell.dt.century())
        +------+
        |    a |
        |  --- |
        |  i32 |
        +======+
        |   20 |
        |   20 |
        |   21 |
        | null |
        +------+
        """

    @abstractmethod
    def date(self) -> Cell[python_datetime.date | None]:
        """
        Extract the date from a datetime.

        Returns
        -------
        cell:
            The date.

        Examples
        --------
        >>> from datetime import datetime
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [datetime(1999, 12, 31), datetime(2000, 1, 1, 12, 30, 0), None])
        >>> column.transform(lambda cell: cell.dt.date())
        +------------+
        | a          |
        | ---        |
        | date       |
        +============+
        | 1999-12-31 |
        | 2000-01-01 |
        | null       |
        +------------+
        """

    @abstractmethod
    def day(self) -> Cell[int | None]:
        """
        Extract the day from a datetime or date.

        Returns
        -------
        cell:
            The day.

        Examples
        --------
        >>> from datetime import datetime, date
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("a", [datetime(1999, 12, 31), datetime(2000, 1, 1), None])
        >>> column1.transform(lambda cell: cell.dt.day())
        +------+
        |    a |
        |  --- |
        |   i8 |
        +======+
        |   31 |
        |    1 |
        | null |
        +------+

        >>> column2 = Column("a", [date(1999, 12, 31), date(2000, 1, 1), None])
        >>> column2.transform(lambda cell: cell.dt.day())
        +------+
        |    a |
        |  --- |
        |   i8 |
        +======+
        |   31 |
        |    1 |
        | null |
        +------+
        """

    @abstractmethod
    def day_of_year(self) -> Cell[int | None]:
        """
        Extract the day of the year from a datetime or date.

        The day of the year is a number between 1 and 366. A 366th day only occurs in leap years.

        Returns
        -------
        cell:
            The day of the year.

        Examples
        --------
        >>> from datetime import datetime, date
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("a", [datetime(1999, 12, 31), datetime(2000, 1, 1), datetime(2000, 12, 31), None])
        >>> column1.transform(lambda cell: cell.dt.day_of_year())
        +------+
        |    a |
        |  --- |
        |  i16 |
        +======+
        |  365 |
        |    1 |
        |  366 |
        | null |
        +------+

        >>> column2 = Column("a", [date(1999, 12, 31), date(2000, 1, 1), date(2000, 12, 31), None])
        >>> column2.transform(lambda cell: cell.dt.day_of_year())
        +------+
        |    a |
        |  --- |
        |  i16 |
        +======+
        |  365 |
        |    1 |
        |  366 |
        | null |
        +------+
        """

    @abstractmethod
    def hour(self) -> Cell[int | None]:
        """
        Extract the hour from a datetime or time.

        Returns
        -------
        cell:
            The hour.

        Examples
        --------
        >>> from datetime import datetime, time
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("a", [datetime(2000, 1, 1, hour=0), datetime(2000, 1, 1, hour=12), None])
        >>> column1.transform(lambda cell: cell.dt.hour())
        +------+
        |    a |
        |  --- |
        |   i8 |
        +======+
        |    0 |
        |   12 |
        | null |
        +------+

        >>> column2 = Column("a", [time(hour=0), time(hour=12), None])
        >>> column2.transform(lambda cell: cell.dt.hour())
        +------+
        |    a |
        |  --- |
        |   i8 |
        +======+
        |    0 |
        |   12 |
        | null |
        +------+
        """

    @abstractmethod
    def microsecond(self) -> Cell[int | None]:
        """
        Extract the microsecond from a datetime or time.

        Returns
        -------
        cell:
            The microsecond.

        Examples
        --------
        >>> from datetime import datetime, time
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("a", [datetime(2000, 1, 1, microsecond=0), datetime(2000, 1, 1, microsecond=500), None])
        >>> column1.transform(lambda cell: cell.dt.microsecond())
        +------+
        |    a |
        |  --- |
        |  i32 |
        +======+
        |    0 |
        |  500 |
        | null |
        +------+

        >>> column2 = Column("a", [time(microsecond=0), time(microsecond=500), None])
        >>> column2.transform(lambda cell: cell.dt.microsecond())
        +------+
        |    a |
        |  --- |
        |  i32 |
        +======+
        |    0 |
        |  500 |
        | null |
        +------+
        """

    @abstractmethod
    def millennium(self) -> Cell[int | None]:
        """
        Extract the millennium from a datetime or date.

        Note that since our calendar begins with year 1 the first millennium lasts from year 1 to year 1000. Subsequent
        centuries begin with years ending in "001" and end with years ending in "000".

        Returns
        -------
        cell:
            The millennium.

        Examples
        --------
        >>> from datetime import datetime, date
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("a", [datetime(1999, 12, 31), datetime(2000, 1, 1), datetime(2001, 1, 1), None])
        >>> column1.transform(lambda cell: cell.dt.millennium())
        +------+
        |    a |
        |  --- |
        |  i32 |
        +======+
        |    2 |
        |    2 |
        |    3 |
        | null |
        +------+

        >>> column2 = Column("a", [date(1999, 12, 31), date(2000, 1, 1), date(2001, 1, 1), None])
        >>> column2.transform(lambda cell: cell.dt.millennium())
        +------+
        |    a |
        |  --- |
        |  i32 |
        +======+
        |    2 |
        |    2 |
        |    3 |
        | null |
        +------+
        """

    @abstractmethod
    def millisecond(self) -> Cell[int | None]:
        """
        Extract the millisecond from a datetime or time.

        Returns
        -------
        cell:
            The millisecond.

        Examples
        --------
        >>> from datetime import datetime, time
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("a", [datetime(2000, 1, 1, microsecond=0), datetime(2000, 1, 1, microsecond=500000), None])
        >>> column1.transform(lambda cell: cell.dt.millisecond())
        +------+
        |    a |
        |  --- |
        |  i32 |
        +======+
        |    0 |
        |  500 |
        | null |
        +------+

        >>> column2 = Column("a", [time(microsecond=0), time(microsecond=500000), None])
        >>> column2.transform(lambda cell: cell.dt.millisecond())
        +------+
        |    a |
        |  --- |
        |  i32 |
        +======+
        |    0 |
        |  500 |
        | null |
        +------+
        """

    @abstractmethod
    def minute(self) -> Cell[int | None]:
        """
        Extract the minute from a datetime or time.

        Returns
        -------
        cell:
            The minute.

        Examples
        --------
        >>> from datetime import datetime, time
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("a", [datetime(2000, 1, 1, minute=0), datetime(2000, 1, 1, minute=30), None])
        >>> column1.transform(lambda cell: cell.dt.minute())
        +------+
        |    a |
        |  --- |
        |   i8 |
        +======+
        |    0 |
        |   30 |
        | null |
        +------+

        >>> column2 = Column("a", [time(minute=0), time(minute=30), None])
        >>> column2.transform(lambda cell: cell.dt.minute())
        +------+
        |    a |
        |  --- |
        |   i8 |
        +======+
        |    0 |
        |   30 |
        | null |
        +------+
        """

    @abstractmethod
    def month(self) -> Cell[int | None]:
        """
        Extract the month from a datetime or date.

        Returns
        -------
        cell:
            The month.

        Examples
        --------
        >>> from datetime import datetime, date
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("a", [datetime(1999, 12, 31), datetime(2000, 1, 1), None])
        >>> column1.transform(lambda cell: cell.dt.month())
        +------+
        |    a |
        |  --- |
        |   i8 |
        +======+
        |   12 |
        |    1 |
        | null |
        +------+

        >>> column2 = Column("a", [date(1999, 12, 31), date(2000, 1, 1), None])
        >>> column2.transform(lambda cell: cell.dt.month())
        +------+
        |    a |
        |  --- |
        |   i8 |
        +======+
        |   12 |
        |    1 |
        | null |
        +------+
        """

    @abstractmethod
    def quarter(self) -> Cell[int | None]:
        """
        Extract the quarter from a datetime or date.

        The quarter is a number between 1 and 4:

        - 1: January to March
        - 2: April to June
        - 3: July to September
        - 4: October to December

        Returns
        -------
        cell:
            The quarter.

        Examples
        --------
        >>> from datetime import datetime, date
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("a", [datetime(1999, 12, 31), datetime(2000, 1, 1), datetime(2000, 4, 1), None])
        >>> column1.transform(lambda cell: cell.dt.quarter())
        +------+
        |    a |
        |  --- |
        |   i8 |
        +======+
        |    4 |
        |    1 |
        |    2 |
        | null |
        +------+

        >>> column2 = Column("a", [date(1999, 12, 31), date(2000, 1, 1), date(2000, 4, 1), None])
        >>> column2.transform(lambda cell: cell.dt.quarter())
        +------+
        |    a |
        |  --- |
        |   i8 |
        +======+
        |    4 |
        |    1 |
        |    2 |
        | null |
        +------+
        """

    @abstractmethod
    def second(self) -> Cell[int | None]:
        """
        Extract the second from a datetime or time.

        Returns
        -------
        cell:
            The second.

        Examples
        --------
        >>> from datetime import datetime, time
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("a", [datetime(2000, 1, 1, second=0), datetime(2000, 1, 1, second=30), None])
        >>> column1.transform(lambda cell: cell.dt.second())
        +------+
        |    a |
        |  --- |
        |   i8 |
        +======+
        |    0 |
        |   30 |
        | null |
        +------+

        >>> column2 = Column("a", [time(second=0), time(second=30), None])
        >>> column2.transform(lambda cell: cell.dt.second())
        +------+
        |    a |
        |  --- |
        |   i8 |
        +======+
        |    0 |
        |   30 |
        | null |
        +------+
        """

    @abstractmethod
    def time(self) -> Cell[python_datetime.time | None]:
        """
        Extract the time from a datetime.

        Returns
        -------
        cell:
            The time.

        Examples
        --------
        >>> from datetime import datetime
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [datetime(1999, 12, 31), datetime(2000, 1, 1, 12, 30, 0), None])
        >>> column.transform(lambda cell: cell.dt.time())
        +----------+
        | a        |
        | ---      |
        | time     |
        +==========+
        | 00:00:00 |
        | 12:30:00 |
        | null     |
        +----------+
        """

    @abstractmethod
    def week(self) -> Cell[int | None]:
        """
        Extract the week from a datetime or date.

        The week is a number between 1 and 53. The first week of a year starts at the first Monday of the year.
        Subsequent weeks start on Monday and end on Sunday.

        Returns
        -------
        cell:
            The week.

        Examples
        --------
        >>> from datetime import datetime, date
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("a", [datetime(1999, 12, 31), datetime(2000, 1, 2), datetime(2000, 1, 3), None])
        >>> column1.transform(lambda cell: cell.dt.week())
        +------+
        |    a |
        |  --- |
        |   i8 |
        +======+
        |   52 |
        |   52 |
        |    1 |
        | null |
        +------+

        >>> column2 = Column("a", [date(1999, 12, 31), date(2000, 1, 2), date(2000, 1, 4), None])
        >>> column2.transform(lambda cell: cell.dt.week())
        +------+
        |    a |
        |  --- |
        |   i8 |
        +======+
        |   52 |
        |   52 |
        |    1 |
        | null |
        +------+
        """

    @abstractmethod
    def weekday(self) -> Cell[int | None]:
        """
        Extract the weekday from a datetime or date.

        The weekday is a number between 1 (Monday) and 7 (Sunday).

        Returns
        -------
        cell:
            The weekday.

        Examples
        --------
        >>> from datetime import datetime, date
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("a", [datetime(2000, 1, 1), datetime(2000, 1, 2), None])
        >>> column1.transform(lambda cell: cell.dt.weekday())
        +------+
        |    a |
        |  --- |
        |   i8 |
        +======+
        |    6 |
        |    7 |
        | null |
        +------+

        >>> column2 = Column("a", [date(2000, 1, 1), date(2000, 1, 2), None])
        >>> column2.transform(lambda cell: cell.dt.weekday())
        +------+
        |    a |
        |  --- |
        |   i8 |
        +======+
        |    6 |
        |    7 |
        | null |
        +------+
        """

    @abstractmethod
    def year(self) -> Cell[int | None]:
        """
        Extract the year from a datetime or date.

        Returns
        -------
        cell:
            The year.

        Examples
        --------
        >>> from datetime import datetime, date
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("a", [datetime(1999, 12, 31), datetime(2000, 1, 1), None])
        >>> column1.transform(lambda cell: cell.dt.year())
        +------+
        |    a |
        |  --- |
        |  i32 |
        +======+
        | 1999 |
        | 2000 |
        | null |
        +------+

        >>> column2 = Column("a", [date(1999, 12, 31), date(2000, 1, 1), None])
        >>> column2.transform(lambda cell: cell.dt.year())
        +------+
        |    a |
        |  --- |
        |  i32 |
        +======+
        | 1999 |
        | 2000 |
        | null |
        +------+
        """

    # ------------------------------------------------------------------------------------------------------------------
    # Other operations
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def is_in_leap_year(self) -> Cell[bool | None]:
        """
        Check a datetime or date is in a leap year.

        Returns
        -------
        cell:
            Whether the year is a leap year.

        Examples
        --------
        >>> from datetime import datetime, date
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("a", [datetime(1900, 1, 1), datetime(2000, 1, 1), None])
        >>> column1.transform(lambda cell: cell.dt.is_in_leap_year())
        +-------+
        | a     |
        | ---   |
        | bool  |
        +=======+
        | false |
        | true  |
        | null  |
        +-------+

        >>> column2 = Column("a", [date(1900, 1, 1), date(2000, 1, 1), None])
        >>> column2.transform(lambda cell: cell.dt.is_in_leap_year())
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

    @abstractmethod
    def replace(
        self,
        *,
        year: _ConvertibleToIntCell = None,
        month: _ConvertibleToIntCell = None,
        day: _ConvertibleToIntCell = None,
        hour: _ConvertibleToIntCell = None,
        minute: _ConvertibleToIntCell = None,
        second: _ConvertibleToIntCell = None,
        microsecond: _ConvertibleToIntCell = None,
    ) -> Cell:
        """
        Replace components of a datetime or date.

        If a component is not provided, it is not changed. Components that are not applicable to the object are ignored,
        e.g. setting the hour of a date. Invalid results are converted to missing values (`None`).

        Parameters
        ----------
        year:
            The new year.
        month:
            The new month. Must be between 1 and 12.
        day:
            The new day. Must be between 1 and 31.
        hour:
            The new hour. Must be between 0 and 23.
        minute:
            The new minute. Must be between 0 and 59.
        second:
            The new second. Must be between 0 and 59.
        microsecond:
            The new microsecond. Must be between 0 and 999999.

        Returns
        -------
        cell:
            The new datetime or date.

        Examples
        --------
        >>> from datetime import datetime, date, time
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("a", [datetime(2000, 1, 1), None])
        >>> column1.transform(lambda cell: cell.dt.replace(month=2, day=2, hour=2))
        +---------------------+
        | a                   |
        | ---                 |
        | datetime[μs]        |
        +=====================+
        | 2000-02-02 02:00:00 |
        | null                |
        +---------------------+

        >>> column2 = Column("a", [date(2000, 1, 1), None])
        >>> column2.transform(lambda cell: cell.dt.replace(month=2, day=2, hour=2))
        +------------+
        | a          |
        | ---        |
        | date       |
        +============+
        | 2000-02-02 |
        | null       |
        +------------+
        """

    # TODO: explain format string + more examples
    @abstractmethod
    def to_string(self, *, format: str = "iso") -> Cell[str | None]:
        """
        Convert a datetime, date, or time to a string.

        The format can be either the special value "iso" to create ISO 8601 strings or a custom format string. The
        custom format string can contain the following placeholders:

        Parameters
        ----------
        format:
            The format to use.

        Returns
        -------
        cell:
            The string representation.

        Raises
        ------
        ValueError
            If the format is invalid.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> import datetime
        >>> column = Column("a", [datetime.date(2022, 1, 9)])
        >>> column.transform(lambda cell: cell.dt.to_string())
        +------------+
        | a          |
        | ---        |
        | str        |
        +============+
        | 2022-01-09 |
        +------------+
        """

    @abstractmethod
    def unix_time(self) -> Cell[int | None]:
        """
        Get the Unix time from a datetime.

        The Unix time is the elapsed time since 00:00:00 UTC on 1 January 1970. This method returns the Unix time in
        microseconds.

        Returns
        -------
        cell:
            The Unix time.

        Examples
        --------
        >>> from datetime import datetime
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [datetime(1970, 1, 1), datetime(1970, 1, 2), None])
        >>> column.transform(lambda cell: cell.dt.unix_time())
        +-------------+
        |           a |
        |         --- |
        |         i64 |
        +=============+
        |           0 |
        | 86400000000 |
        |        null |
        +-------------+
        """
