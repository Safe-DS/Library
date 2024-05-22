from __future__ import annotations

import datetime
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
    >>> column = Column("example", ["2021-01-01", "2021-02-01", "abc"])
    >>> column.transform(lambda cell: cell.str.to_string())
    +------------+
    | example    |
    | ---        |
    | date       |
    +============+
    | 2021-01-01 |
    | 2021-02-01 |
    | null       |
    +------------+
    """

    @abstractmethod
    def datetime_to_string(self) -> Cell[str | None]:
        """
        Convert the date value in the cell to a string.

        Parameters
        ----------

        Returns
        -------
        date:
            The string value. If the date cannot be converted to a date, None is returned.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", ["2021-01-01", "2021-02-01", "abc"])
        >>> column.transform(lambda cell: cell.str.to_string())
        +------------+
        | example    |
        | ---        |
        | date       |
        +============+
        | 2021-01-01 |
        | 2021-02-01 |
        | null       |
        +------------+
        """

    @abstractmethod
    def date_to_string(self) -> Cell[str | None]:
        """
        Convert the date value in the cell to a string.

        Parameters
        ----------

        Returns
        -------
        date:
            The string value. If the date cannot be converted to a date, None is returned.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", ["2021-01-01", "2021-02-01", "abc"])
        >>> column.transform(lambda cell: cell.str.to_string())
        +------------+
        | example    |
        | ---        |
        | date       |
        +============+
        | 2021-01-01 |
        | 2021-02-01 |
        | null       |
        +------------+
        """
