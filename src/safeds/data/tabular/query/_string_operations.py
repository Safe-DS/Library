from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import datetime

    from safeds._typing import _ConvertibleToIntCell, _ConvertibleToStringCell
    from safeds.data.tabular.containers import Cell

# TODO: examples with None
# TODO: add more methods
#  - reverse
#  - to_time
#  - ...


class StringOperations(ABC):
    """
    Namespace for operations on strings.

    This class cannot be instantiated directly. It can only be accessed using the `str` attribute of a cell.

    Examples
    --------
    >>> from safeds.data.tabular.containers import Column
    >>> column = Column("a", ["ab", "bc", "cd"])
    >>> column.transform(lambda cell: cell.str.to_uppercase())
    +-----+
    | a   |
    | --- |
    | str |
    +=====+
    | AB  |
    | BC  |
    | CD  |
    +-----+
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
    # String operations
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def contains(self, substring: _ConvertibleToStringCell) -> Cell[bool | None]:
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
        >>> column = Column("a", ["ab", "bc", "cd", None])
        >>> column.transform(lambda cell: cell.str.contains("b"))
        +-------+
        | a     |
        | ---   |
        | bool  |
        +=======+
        | true  |
        | true  |
        | false |
        | null  |
        +-------+
        """

    @abstractmethod
    def ends_with(self, suffix: _ConvertibleToStringCell) -> Cell[bool | None]:
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
        >>> column = Column("a", ["ab", "bc", "cd", None])
        >>> column.transform(lambda cell: cell.str.ends_with("c"))
        +-------+
        | a     |
        | ---   |
        | bool  |
        +=======+
        | false |
        | true  |
        | false |
        | null  |
        +-------+
        """

    @abstractmethod
    def index_of(self, substring: _ConvertibleToStringCell) -> Cell[int | None]:
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
        >>> column = Column("a", ["ab", "bc", "cd", None])
        >>> column.transform(lambda cell: cell.str.index_of("b"))
        +------+
        |    a |
        |  --- |
        |  u32 |
        +======+
        |    1 |
        |    0 |
        | null |
        | null |
        +------+
        """

    @abstractmethod
    def length(self, *, optimize_for_ascii: bool = False) -> Cell[int | None]:
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
        >>> column = Column("a", ["", "a", "abc", None])
        >>> column.transform(lambda cell: cell.str.length())
        +------+
        |    a |
        |  --- |
        |  u32 |
        +======+
        |    0 |
        |    1 |
        |    3 |
        | null |
        +------+
        """

    @abstractmethod
    def replace(self, old: _ConvertibleToStringCell, new: _ConvertibleToStringCell) -> Cell[str | None]:
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
        >>> column = Column("a", ["ab", "bc", "cd", None])
        >>> column.transform(lambda cell: cell.str.replace("b", "z"))
        +------+
        | a    |
        | ---  |
        | str  |
        +======+
        | az   |
        | zc   |
        | cd   |
        | null |
        +------+
        """

    @abstractmethod
    def starts_with(self, prefix: _ConvertibleToStringCell) -> Cell[bool | None]:
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
        >>> column = Column("a", ["ab", "bc", "cd", None])
        >>> column.transform(lambda cell: cell.str.starts_with("a"))
        +-------+
        | a     |
        | ---   |
        | bool  |
        +=======+
        | true  |
        | false |
        | false |
        | null  |
        +-------+
        """

    @abstractmethod
    def substring(
        self,
        *,
        start: _ConvertibleToIntCell = 0,
        length: _ConvertibleToIntCell = None,
    ) -> Cell[str | None]:
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
        >>> column = Column("a", ["abc", "def", "ghi", None])
        >>> column.transform(lambda cell: cell.str.substring(start=1, length=2))
        +------+
        | a    |
        | ---  |
        | str  |
        +======+
        | bc   |
        | ef   |
        | hi   |
        | null |
        +------+
        """

    # TODO: add format parameter + document
    @abstractmethod
    def to_date(self, *, format: str | None = None) -> Cell[datetime.date | None]:
        """
        Convert the string value in the cell to a date.

        Returns
        -------
        date:
            The date value. If the string cannot be converted to a date, None is returned.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", ["2021-01-01", "2021-02-01", "abc", None])
        >>> column.transform(lambda cell: cell.str.to_date())
        +------------+
        | a          |
        | ---        |
        | date       |
        +============+
        | 2021-01-01 |
        | 2021-02-01 |
        | null       |
        | null       |
        +------------+
        """

    # TODO: add format parameter + document
    @abstractmethod
    def to_datetime(self, *, format: str | None = None) -> Cell[datetime.datetime | None]:
        """
        Convert the string value in the cell to a datetime.

        Returns
        -------
        datetime:
            The datetime value. If the string cannot be converted to a datetime, None is returned.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", ["2021-01-01T00:00:00Z", "2021-02-01T00:00:00Z", "abc", None])
        >>> column.transform(lambda cell: cell.str.to_datetime())
        +-------------------------+
        | a                       |
        | ---                     |
        | datetime[μs, UTC]       |
        +=========================+
        | 2021-01-01 00:00:00 UTC |
        | 2021-02-01 00:00:00 UTC |
        | null                    |
        | null                    |
        +-------------------------+
        """

    # TODO: add to_time

    @abstractmethod
    def to_int(self, *, base: _ConvertibleToIntCell = 10) -> Cell[int | None]:
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
        >>> column1 = Column("a", ["1", "2", "3", "abc", None])
        >>> column1.transform(lambda cell: cell.str.to_int())
        +------+
        |    a |
        |  --- |
        |  i64 |
        +======+
        |    1 |
        |    2 |
        |    3 |
        | null |
        | null |
        +------+

        >>> column2 = Column("a", ["1", "10", "11", "abc", None])
        >>> column2.transform(lambda cell: cell.str.to_int(base=2))
        +------+
        |    a |
        |  --- |
        |  i64 |
        +======+
        |    1 |
        |    2 |
        |    3 |
        | null |
        | null |
        +------+
        """

    @abstractmethod
    def to_lowercase(self) -> Cell[str | None]:
        """
        Convert the string value in the cell to lowercase.

        Returns
        -------
        lowercase:
            The string value in lowercase.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", ["AB", "BC", "CD", None])
        >>> column.transform(lambda cell: cell.str.to_lowercase())
        +------+
        | a    |
        | ---  |
        | str  |
        +======+
        | ab   |
        | bc   |
        | cd   |
        | null |
        +------+
        """

    @abstractmethod
    def to_uppercase(self) -> Cell[str | None]:
        """
        Convert the string value in the cell to uppercase.

        Returns
        -------
        uppercase:
            The string value in uppercase.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", ["ab", "bc", "cd", None])
        >>> column.transform(lambda cell: cell.str.to_uppercase())
        +------+
        | a    |
        | ---  |
        | str  |
        +======+
        | AB   |
        | BC   |
        | CD   |
        | null |
        +------+
        """

    @abstractmethod
    def trim(self) -> Cell[str | None]:
        """
        Remove whitespace from the start and end of the string value in the cell.

        Returns
        -------
        trimmed:
            The string value without whitespace at the start and end.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", ["", " abc", "abc ", " abc ", None])
        >>> column.transform(lambda cell: cell.str.trim())
        +------+
        | a    |
        | ---  |
        | str  |
        +======+
        |      |
        | abc  |
        | abc  |
        | abc  |
        | null |
        +------+
        """

    @abstractmethod
    def trim_end(self) -> Cell[str | None]:
        """
        Remove whitespace from the end of the string value in the cell.

        Returns
        -------
        trimmed:
            The string value without whitespace at the end.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", ["", " abc", "abc ", " abc ", None])
        >>> column.transform(lambda cell: cell.str.trim_end())
        +------+
        | a    |
        | ---  |
        | str  |
        +======+
        |      |
        |  abc |
        | abc  |
        |  abc |
        | null |
        +------+
        """

    @abstractmethod
    def trim_start(self) -> Cell[str | None]:
        """
        Remove whitespace from the start of the string value in the cell.

        Returns
        -------
        trimmed:
            The string value without whitespace at the start.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", ["", " abc", "abc ", " abc ", None])
        >>> column.transform(lambda cell: cell.str.trim_start())
        +------+
        | a    |
        | ---  |
        | str  |
        +======+
        |      |
        | abc  |
        | abc  |
        | abc  |
        | null |
        +------+
        """
