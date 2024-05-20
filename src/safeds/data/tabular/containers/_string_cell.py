from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import datetime

    from safeds.data.tabular.containers import Cell


class StringCell(ABC):
    """
    Namespace for operations on strings.

    This class cannot be instantiated directly. It can only be accessed using the `str` attribute of a cell.

    Examples
    --------
    >>> from safeds.data.tabular.containers import Column
    >>> column = Column("example", ["ab", "bc", "cd"])
    >>> column.transform(lambda cell: cell.str.to_uppercase())
    +---------+
    | example |
    | ---     |
    | str     |
    +=========+
    | AB      |
    | BC      |
    | CD      |
    +---------+
    """

    @abstractmethod
    def contains(self, substring: str) -> Cell[bool]:
        """
        Check if the string value in the cell contains the substring.

        Parameters
        ----------
        substring:
            The substring to search for.

        Returns
        -------
        contains:
            Whether the string value contains the substring.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", ["ab", "bc", "cd"])
        >>> column.count_if(lambda cell: cell.str.contains("b"))
        2
        """

    @abstractmethod
    def ends_with(self, suffix: str) -> Cell[bool]:
        """
        Check if the string value in the cell ends with the suffix.

        Parameters
        ----------
        suffix:
            The suffix to search for.

        Returns
        -------
        ends_with:
            Whether the string value ends with the suffix.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", ["ab", "bc", "cd"])
        >>> column.count_if(lambda cell: cell.str.ends_with("c"))
        1
        """

    @abstractmethod
    def index_of(self, substring: str) -> Cell[int | None]:
        """
        Get the index of the first occurrence of the substring in the string value in the cell.

        Parameters
        ----------
        substring:
            The substring to search for.

        Returns
        -------
        index_of:
            The index of the first occurrence of the substring. If the substring is not found, None is returned.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", ["ab", "bc", "cd"])
        >>> column.transform(lambda cell: cell.str.index_of("b"))
        +---------+
        | example |
        |     --- |
        |     u32 |
        +=========+
        |       1 |
        |       0 |
        |    null |
        +---------+
        """

    @abstractmethod
    def length(self, *, optimize_for_ascii: bool = False) -> Cell[int]:
        """
        Get the number of characters of the string value in the cell.

        Parameters
        ----------
        optimize_for_ascii:
            Greatly speed up this operation if the string is ASCII-only. If the string contains non-ASCII characters,
            this option will return incorrect results, though.

        Returns
        -------
        length:
            The length of the string value.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", ["", "a", "abc"])
        >>> column.transform(lambda cell: cell.str.length())
        +---------+
        | example |
        |     --- |
        |     u32 |
        +=========+
        |       0 |
        |       1 |
        |       3 |
        +---------+
        """

    @abstractmethod
    def replace(self, old: str, new: str) -> Cell[str]:
        """
        Replace occurrences of the old substring with the new substring in the string value in the cell.

        Parameters
        ----------
        old:
            The substring to replace.
        new:
            The substring to replace with.

        Returns
        -------
        replaced_string:
            The string value with the occurrences replaced.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", ["ab", "bc", "cd"])
        >>> column.transform(lambda cell: cell.str.replace("b", "z"))
        +---------+
        | example |
        | ---     |
        | str     |
        +=========+
        | az      |
        | zc      |
        | cd      |
        +---------+
        """

    @abstractmethod
    def starts_with(self, prefix: str) -> Cell[bool]:
        """
        Check if the string value in the cell starts with the prefix.

        Parameters
        ----------
        prefix:
            The prefix to search for.

        Returns
        -------
        starts_with:
            Whether the string value starts with the prefix.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", ["ab", "bc", "cd"])
        >>> column.count_if(lambda cell: cell.str.starts_with("a"))
        1
        """

    @abstractmethod
    def substring(self, start: int = 0, length: int | None = None) -> Cell[str]:
        """
        Get a substring of the string value in the cell.

        Parameters
        ----------
        start:
            The start index of the substring.
        length:
            The length of the substring. If None, the slice contains all rows starting from `start`. Must greater than
            or equal to 0.

        Returns
        -------
        substring:
            The substring of the string value.

        Raises
        ------
        OutOfBoundsError
            If length is less than 0.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", ["abc", "def", "ghi"])
        >>> column.transform(lambda cell: cell.str.substring(1, 2))
        +---------+
        | example |
        | ---     |
        | str     |
        +=========+
        | bc      |
        | ef      |
        | hi      |
        +---------+
        """

    @abstractmethod
    def to_date(self) -> Cell[datetime.date | None]:
        """
        Convert the string value in the cell to a date. Requires the string to be in the ISO 8601 format.

        Returns
        -------
        date:
            The date value. If the string cannot be converted to a date, None is returned.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", ["2021-01-01", "2021-02-01", "abc"])
        >>> column.transform(lambda cell: cell.str.to_date())
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
    def to_datetime(self) -> Cell[datetime.datetime | None]:
        """
        Convert the string value in the cell to a datetime. Requires the string to be in the ISO 8601 format.

        Returns
        -------
        datetime:
            The datetime value. If the string cannot be converted to a datetime, None is returned.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", ["2021-01-01T00:00:00z", "2021-02-01T00:00:00z", "abc"])
        >>> column.transform(lambda cell: cell.str.to_datetime())
        +-------------------------+
        | example                 |
        | ---                     |
        | datetime[μs, UTC]       |
        +=========================+
        | 2021-01-01 00:00:00 UTC |
        | 2021-02-01 00:00:00 UTC |
        | null                    |
        +-------------------------+
        """

    @abstractmethod
    def to_float(self) -> Cell[float | None]:
        """
        Convert the string value in the cell to a float.

        Returns
        -------
        float:
            The float value. If the string cannot be converted to a float, None is returned.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", ["1", "3.4", "5.6", "abc"])
        >>> column.transform(lambda cell: cell.str.to_float())
        +---------+
        | example |
        |     --- |
        |     f64 |
        +=========+
        | 1.00000 |
        | 3.40000 |
        | 5.60000 |
        |    null |
        +---------+
        """

    @abstractmethod
    def to_int(self, *, base: int = 10) -> Cell[int | None]:
        """
        Convert the string value in the cell to an integer.

        Parameters
        ----------
        base:
            The base of the integer.

        Returns
        -------
        int:
            The integer value. If the string cannot be converted to an integer, None is returned.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("example", ["1", "2", "3", "abc"])
        >>> column1.transform(lambda cell: cell.str.to_int())
        +---------+
        | example |
        |     --- |
        |     i64 |
        +=========+
        |       1 |
        |       2 |
        |       3 |
        |    null |
        +---------+

        >>> column2 = Column("example", ["1", "10", "11", "abc"])
        >>> column2.transform(lambda cell: cell.str.to_int(base=2))
        +---------+
        | example |
        |     --- |
        |     i64 |
        +=========+
        |       1 |
        |       2 |
        |       3 |
        |    null |
        +---------+
        """

    @abstractmethod
    def to_lowercase(self) -> Cell[str]:
        """
        Convert the string value in the cell to lowercase.

        Returns
        -------
        lowercase:
            The string value in lowercase.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", ["AB", "BC", "CD"])
        >>> column.transform(lambda cell: cell.str.to_lowercase())
        +---------+
        | example |
        | ---     |
        | str     |
        +=========+
        | ab      |
        | bc      |
        | cd      |
        +---------+
        """

    @abstractmethod
    def to_uppercase(self) -> Cell[str]:
        """
        Convert the string value in the cell to uppercase.

        Returns
        -------
        uppercase:
            The string value in uppercase.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", ["ab", "bc", "cd"])
        >>> column.transform(lambda cell: cell.str.to_uppercase())
        +---------+
        | example |
        | ---     |
        | str     |
        +=========+
        | AB      |
        | BC      |
        | CD      |
        +---------+
        """

    @abstractmethod
    def trim(self) -> Cell[str]:
        """
        Remove whitespace from the start and end of the string value in the cell.

        Returns
        -------
        trimmed:
            The string value without whitespace at the start and end.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", ["", " abc", "abc ", " abc "])
        >>> column.transform(lambda cell: cell.str.trim())
        +---------+
        | example |
        | ---     |
        | str     |
        +=========+
        |         |
        | abc     |
        | abc     |
        | abc     |
        +---------+
        """

    @abstractmethod
    def trim_end(self) -> Cell[str]:
        """
        Remove whitespace from the end of the string value in the cell.

        Returns
        -------
        trimmed:
            The string value without whitespace at the end.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", ["", " abc", "abc ", " abc "])
        >>> column.transform(lambda cell: cell.str.trim_end())
        +---------+
        | example |
        | ---     |
        | str     |
        +=========+
        |         |
        |  abc    |
        | abc     |
        |  abc    |
        +---------+
        """

    @abstractmethod
    def trim_start(self) -> Cell[str]:
        """
        Remove whitespace from the start of the string value in the cell.

        Returns
        -------
        trimmed:
            The string value without whitespace at the start.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", ["", " abc", "abc ", " abc "])
        >>> column.transform(lambda cell: cell.str.trim_start())
        +---------+
        | example |
        | ---     |
        | str     |
        +=========+
        |         |
        | abc     |
        | abc     |
        | abc     |
        +---------+
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
