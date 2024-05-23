from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from safeds.data.tabular.containers import Cell


class TemporalCell(ABC):
    """
    A class that contains temporal methods for a column.

    Parameters
    ----------
    column:
        The column to be operated on.

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
    def datetime_to_string(self, format_string: str = "%Y/%m/%d %H:%M:%S") -> Cell[str | None]:
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
    def date_to_string(self, format_string: str = "%F") -> Cell[str | None]:
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
