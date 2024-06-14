from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from safeds.data.tabular.containers import Cell


class TemporalCell(ABC):
    """
    Namespace for operations on temporal data.

    This class cannot be instantiated directly. It can only be accessed using the `dt` attribute of a cell.

    Examples
    --------
    >>> from safeds.data.tabular.containers import Column
    >>> import datetime
    >>> column = Column("example", [datetime.date(2022, 1, 9)])
    >>> column.transform(lambda cell: cell.dt.date_to_string("%Y/%m/%d"))
    +------------+
    | example    |
    | ---        |
    | str        |
    +============+
    | 2022/01/09 |
    +------------+
    """

    @abstractmethod
    def century(self) -> Cell[int]:
        """
        Get the century of the underlying date(time) data.

        Returns
        -------
            A cell containing the century as integer.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> import datetime
        >>> column = Column("example", [datetime.date(2022, 1, 1)])
        >>> column.transform(lambda cell: cell.dt.century())
        +---------+
        | example |
        |     --- |
        |     i32 |
        +=========+
        |      21 |
        +---------+
        """

    @abstractmethod
    def weekday(self) -> Cell[int]:
        """
        Get the weekday of the underlying date(time) data.

        Returns
        -------
            A cell containing the weekday as integer.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> import datetime
        >>> column = Column("example", [datetime.date(2022, 1, 1)])
        >>> column.transform(lambda cell: cell.dt.weekday())
        +---------+
        | example |
        |     --- |
        |      i8 |
        +=========+
        |       6 |
        +---------+
        """

    @abstractmethod
    def week(self) -> Cell[int]:
        """
        Get the week of the underlying date(time) data.

        Returns
        -------
            A cell containing the week as integer.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> import datetime
        >>> column = Column("example", [datetime.date(2022, 1, 1)])
        >>> column.transform(lambda cell: cell.dt.week())
        +---------+
        | example |
        |     --- |
        |      i8 |
        +=========+
        |      52 |
        +---------+
        """

    @abstractmethod
    def year(self) -> Cell[int]:
        """
        Get the year of the underlying date(time) data.

        Returns
        -------
            A cell containing the year as integer.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> import datetime
        >>> column = Column("example", [datetime.date(2022, 1, 9)])
        >>> column.transform(lambda cell: cell.dt.year())
        +---------+
        | example |
        |     --- |
        |     i32 |
        +=========+
        |    2022 |
        +---------+
        """

    @abstractmethod
    def month(self) -> Cell[int]:
        """
        Get the month of the underlying date(time) data.

        Returns
        -------
            A cell containing the month as integer.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> import datetime
        >>> column = Column("example", [datetime.date(2022, 1, 9)])
        >>> column.transform(lambda cell: cell.dt.month())
        +---------+
        | example |
        |     --- |
        |      i8 |
        +=========+
        |       1 |
        +---------+
        """

    @abstractmethod
    def day(self) -> Cell[int]:
        """
        Get the day of the underlying date(time) data.

        Returns
        -------
            A cell containing the day as integer.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> import datetime
        >>> column = Column("example", [datetime.date(2022, 1, 9)])
        >>> column.transform(lambda cell: cell.dt.day())
        +---------+
        | example |
        |     --- |
        |      i8 |
        +=========+
        |       9 |
        +---------+
        """

    @abstractmethod
    def datetime_to_string(self, format_string: str = "%Y/%m/%d %H:%M:%S") -> Cell[str]:
        """
        Convert the date value in the cell to a string.

        Parameters
        ----------
        format_string:
            The format string it will be used to convert the data into the string.

        Returns
        -------
        date:
            The string value.

        Raises
        ------
        ValueError
            If the formatstring is invalid.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> import datetime
        >>> column = Column("example", [ datetime.datetime(2022, 1, 9, 23, 29, 1, tzinfo=datetime.UTC)])
        >>> column.transform(lambda cell: cell.dt.datetime_to_string())
        +---------------------+
        | example             |
        | ---                 |
        | str                 |
        +=====================+
        | 2022/01/09 23:29:01 |
        +---------------------+
        """

    @abstractmethod
    def date_to_string(self, format_string: str = "%F") -> Cell[str]:
        """
        Convert the date value in the cell to a string.

        Parameters
        ----------
        format_string:
            The format string it will be used to convert the data into the string.

        Returns
        -------
        date:
            The string value.


        ValueError
            If the formatstring is invalid.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> import datetime
        >>> column = Column("example", [datetime.date(2022, 1, 9)])
        >>> column.transform(lambda cell: cell.dt.date_to_string())
        +------------+
        | example    |
        | ---        |
        | str        |
        +============+
        | 2022-01-09 |
        +------------+
        """

    # ------------------------------------------------------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def _equals(self, other: object) -> bool:
        """
        Check if this cell is equal to another object.

        This method is needed because the `__eq__` method is used for element-wise comparisons.
        """
