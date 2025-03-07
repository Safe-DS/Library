from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from datetime import timedelta

    from safeds.data.tabular.containers import Cell


class DurationOperations(ABC):
    """
    Namespace for operations on durations.

    This class cannot be instantiated directly. It can only be accessed using the `dur` attribute of a cell.

    Examples
    --------
    >>> from datetime import timedelta
    >>> from safeds.data.tabular.containers import Column
    >>> column = Column("a", [timedelta(days=-1), timedelta(days=0), timedelta(days=1)])
    >>> column.transform(lambda cell: cell.dur.abs())
    +--------------+
    | a            |
    | ---          |
    | duration[μs] |
    +==============+
    | 1d           |
    | 0µs          |
    | 1d           |
    +--------------+
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
    # Duration operations
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def abs(self) -> Cell[timedelta | None]:
        """
        Get the absolute value of the duration.

        Returns
        -------
        cell:
            The absolute value.

        Examples
        --------
        >>> from datetime import timedelta
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [timedelta(days=-1), timedelta(days=1), None])
        >>> column.transform(lambda cell: cell.dur.abs())
        +--------------+
        | a            |
        | ---          |
        | duration[μs] |
        +==============+
        | 1d           |
        | 1d           |
        | null         |
        +--------------+
        """

    @abstractmethod
    def full_weeks(self) -> Cell[int | None]:
        """
        Get the number of full weeks in the duration. The result is rounded toward zero.

        Returns
        -------
        cell:
            The number of full weeks.

        Examples
        --------
        >>> from datetime import timedelta
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [timedelta(days=8), timedelta(days=6), None])
        >>> column.transform(lambda cell: cell.dur.full_weeks())
        +------+
        |    a |
        |  --- |
        |  i64 |
        +======+
        |    1 |
        |    0 |
        | null |
        +------+
        """

    @abstractmethod
    def full_days(self) -> Cell[int | None]:
        """
        Get the number of full days in the duration. The result is rounded toward zero.

        Returns
        -------
        cell:
            The number of full days.

        Examples
        --------
        >>> from datetime import timedelta
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [timedelta(hours=25), timedelta(hours=23), None])
        >>> column.transform(lambda cell: cell.dur.full_days())
        +------+
        |    a |
        |  --- |
        |  i64 |
        +======+
        |    1 |
        |    0 |
        | null |
        +------+
        """

    @abstractmethod
    def full_hours(self) -> Cell[int | None]:
        """
        Get the number of full hours in the duration. The result is rounded toward zero.

        Returns
        -------
        cell:
            The number of full hours.

        Examples
        --------
        >>> from datetime import timedelta
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [timedelta(minutes=61), timedelta(minutes=59), None])
        >>> column.transform(lambda cell: cell.dur.full_hours())
        +------+
        |    a |
        |  --- |
        |  i64 |
        +======+
        |    1 |
        |    0 |
        | null |
        +------+
        """

    @abstractmethod
    def full_minutes(self) -> Cell[int | None]:
        """
        Get the number of full minutes in the duration. The result is rounded toward zero.

        Returns
        -------
        cell:
            The number of full minutes.

        Examples
        --------
        >>> from datetime import timedelta
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [timedelta(seconds=61), timedelta(seconds=59), None])
        >>> column.transform(lambda cell: cell.dur.full_minutes())
        +------+
        |    a |
        |  --- |
        |  i64 |
        +======+
        |    1 |
        |    0 |
        | null |
        +------+
        """

    @abstractmethod
    def full_seconds(self) -> Cell[int | None]:
        """
        Get the number of full seconds in the duration. The result is rounded toward zero.

        Returns
        -------
        cell:
            The number of full seconds.

        Examples
        --------
        >>> from datetime import timedelta
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [timedelta(milliseconds=1001), timedelta(milliseconds=999), None])
        >>> column.transform(lambda cell: cell.dur.full_seconds())
        +------+
        |    a |
        |  --- |
        |  i64 |
        +======+
        |    1 |
        |    0 |
        | null |
        +------+
        """

    @abstractmethod
    def full_milliseconds(self) -> Cell[int | None]:
        """
        Get the number of full milliseconds in the duration. The result is rounded toward zero.

        Returns
        -------
        cell:
            The number of full milliseconds.

        Examples
        --------
        >>> from datetime import timedelta
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [timedelta(microseconds=1001), timedelta(microseconds=999), None])
        >>> column.transform(lambda cell: cell.dur.full_milliseconds())
        +------+
        |    a |
        |  --- |
        |  i64 |
        +======+
        |    1 |
        |    0 |
        | null |
        +------+
        """

    @abstractmethod
    def full_microseconds(self) -> Cell[int | None]:
        """
        Get the number of full microseconds in the duration. The result is rounded toward zero.

        Since durations only have microsecond resolution at the moment, the rounding has no effect. This may change in
        the future.

        Returns
        -------
        cell:
            The number of full microseconds.

        Examples
        --------
        >>> from datetime import timedelta
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", [timedelta(microseconds=1001), timedelta(microseconds=999), None])
        >>> column.transform(lambda cell: cell.dur.full_microseconds())
        +------+
        |    a |
        |  --- |
        |  i64 |
        +======+
        | 1001 |
        |  999 |
        | null |
        +------+
        """

    @abstractmethod
    def to_string(
        self,
        *,
        format: Literal["iso", "pretty"] = "iso",
    ) -> Cell[str | None]:
        """
        Convert the duration to a string.

        The following formats are supported:

        - `"iso"`: The duration is represented in the ISO 8601 format. This is the default.
        - `"pretty"`: The duration is represented in a human-readable format.

        !!! warning "API Stability"

            Do not rely on the exact output of the `"pretty"` format. In future versions, we may change it without prior
            notice.

        Parameters
        ----------
        format:
            The format to use.

        Returns
        -------
        cell:
            The string representation.
        """
