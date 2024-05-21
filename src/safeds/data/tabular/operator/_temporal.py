from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from safeds.data.tabular.containers import Column


class Temporal:
    """
    A class that contains temporal methods for a column.

    Parameters
    ----------
    column:
        The column to be operated on.

    Examples
    --------

    """

    def __init__(self, column: Column):
        self._column = column

    def from_string(self, format_string: str) -> Column:
        """
        Return a new column with the string values converted to temporal data.

        Parameters
        ----------
        format_string :
            The used format string to convert the string into operator data.

        Returns
        -------
        transformed_column:
            A new column with temporal data instead of the string.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("dates", ["01:01:2021", "01:01:2022", "01:01:2023", "01:01:2024"])
        >>> column.temporal.from_string("%d:%m:%Y")
        | dates      |
        | ---        |
        | date       |
        +============+
        | 2021-01-01 |
        | 2022-01-01 |
        | 2023-01-01 |
        | 2024-01-01 |
        +------------+
        """
        from polars import Date
        from safeds.data.tabular.containers import Column

        temp = self._column._series.str.strptime(Date, format_string)

        return Column._from_polars_series(temp)

    def to_string(self, format_string: str) -> Column:
        """
        Return a new column with the temporal values converted to string data.

        Parameters
        ----------
        format_string :
            The used format string to convert the temporal data into string data.

        Returns
        -------
        transformed_column:
            A new column with string data instead of the temporal data.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("dates", ["01:01:2021", "01:01:2022", "01:01:2023", "01:01:2024"])
        >>> column = column.temporal.from_string("%d:%m:%Y")
        >>> column.temporal.to_string("%Y/%m:%d")
        | dates      |
        | ---        |
        | str       |
        +============+
        | "2021/01/01" |
        | "2022/01/01" |
        | "2023/01/01" |
        | "2024/01/01" |
        +------------+
        """
        from safeds.data.tabular.containers import Column

        return Column._from_polars_series(self._column._series.dt.to_string(format_string))

