from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from safeds._typing import _ConvertibleToIntCell, _ConvertibleToStringCell
    from safeds.data.tabular.containers import Cell
    from safeds.exceptions import OutOfBoundsError  # noqa: F401


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
    def ends_with(self, suffix: _ConvertibleToStringCell) -> Cell[bool | None]:
        """
        Check if the string ends with the suffix.

        Parameters
        ----------
        suffix:
            The expected suffix.

        Returns
        -------
        cell:
            Whether the string ends with the suffix.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", ["ab", "bc", None])
        >>> column.transform(lambda cell: cell.str.ends_with("b"))
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

    @abstractmethod
    def length(self, *, optimize_for_ascii: bool = False) -> Cell[int | None]:
        """
        Get the number of characters.

        Parameters
        ----------
        optimize_for_ascii:
            Greatly speed up this operation if the string is ASCII-only. If the string contains non-ASCII characters,
            this option will return incorrect results, though.

        Returns
        -------
        cell:
            The number of characters.

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
    def pad_end(self, length: int, *, character: str = " ") -> Cell[str | None]:
        """
        Pad the end of the string with the given character until it has the given length.

        Parameters
        ----------
        length:
            The minimum length of the string. If the string is already at least as long, it is returned unchanged. Must
            be greater than or equal to 0.
        character:
            How to pad the string. Must be a single character.

        Returns
        -------
        cell:
            The padded string.

        Raises
        ------
        OutOfBoundsError
            If `length` is less than 0.
        ValueError
            If `char` is not a single character.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", ["ab", "bcde", None])
        >>> column.transform(lambda cell: cell.str.pad_end(3))
        +------+
        | a    |
        | ---  |
        | str  |
        +======+
        | ab   |
        | bcde |
        | null |
        +------+

        >>> column.transform(lambda cell: cell.str.pad_end(3, character="~"))
        +------+
        | a    |
        | ---  |
        | str  |
        +======+
        | ab~  |
        | bcde |
        | null |
        +------+
        """

    @abstractmethod
    def pad_start(self, length: int, *, character: str = " ") -> Cell[str | None]:
        """
        Pad the start of the string with the given character until it has the given length.

        Parameters
        ----------
        length:
            The minimum length of the string. If the string is already at least as long, it is returned unchanged. Must
            be greater than or equal to 0.
        character:
            How to pad the string. Must be a single character.

        Returns
        -------
        cell:
            The padded string.

        Raises
        ------
        OutOfBoundsError
            If `length` is less than 0.
        ValueError
            If `char` is not a single character.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", ["ab", "bcde", None])
        >>> column.transform(lambda cell: cell.str.pad_start(3))
        +------+
        | a    |
        | ---  |
        | str  |
        +======+
        |  ab  |
        | bcde |
        | null |
        +------+

        >>> column.transform(lambda cell: cell.str.pad_start(3, character="~"))
        +------+
        | a    |
        | ---  |
        | str  |
        +======+
        | ~ab  |
        | bcde |
        | null |
        +------+
        """

    @abstractmethod
    def repeat(self, count: _ConvertibleToIntCell) -> Cell[str | None]:
        """
        Repeat the string a number of times.

        Parameters
        ----------
        count:
            The number of times to repeat the string. Must be greater than or equal to 0.

        Returns
        -------
        cell:
            The repeated string.

        Raises
        ------
        OutOfBoundsError
            If `count` is less than 0.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", ["ab", "bc", None])
        >>> column.transform(lambda cell: cell.str.repeat(2))
        +------+
        | a    |
        | ---  |
        | str  |
        +======+
        | abab |
        | bcbc |
        | null |
        +------+
        """

    @abstractmethod
    def reverse(self) -> Cell[str | None]:
        """
        Reverse the string.

        Returns
        -------
        cell:
            The reversed string.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", ["ab", "bc", None])
        >>> column.transform(lambda cell: cell.str.reverse())
        +------+
        | a    |
        | ---  |
        | str  |
        +======+
        | ba   |
        | cb   |
        | null |
        +------+
        """

    @abstractmethod
    def slice(
        self,
        *,
        start: _ConvertibleToIntCell = 0,
        length: _ConvertibleToIntCell = None,
    ) -> Cell[str | None]:
        """
        Get a slice of the string.

        Parameters
        ----------
        start:
            The start index of the slice. Nonnegative indices are counted from the beginning (starting at 0), negative
            indices from the end (starting at -1).
        length:
            The length of the slice. If None, the slice contains all characters starting from `start`. Must greater than
            or equal to 0.

        Returns
        -------
        cell:
            The sliced string.

        Raises
        ------
        OutOfBoundsError
            If `length` is less than 0.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", ["abc", "de", None])
        >>> column.transform(lambda cell: cell.str.slice(start=1))
        +------+
        | a    |
        | ---  |
        | str  |
        +======+
        | bc   |
        | e    |
        | null |
        +------+

        >>> column.transform(lambda cell: cell.str.slice(start=1, length=1))
        +------+
        | a    |
        | ---  |
        | str  |
        +======+
        | b    |
        | e    |
        | null |
        +------+
        """

    @abstractmethod
    def starts_with(self, prefix: _ConvertibleToStringCell) -> Cell[bool | None]:
        """
        Check if the string starts with the prefix.

        Parameters
        ----------
        prefix:
            The expected prefix.

        Returns
        -------
        cell:
            Whether the string starts with the prefix.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", ["ab", "bc", None])
        >>> column.transform(lambda cell: cell.str.starts_with("a"))
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

    @abstractmethod
    def strip(self, *, characters: _ConvertibleToStringCell = None) -> Cell[str | None]:
        """
        Remove leading and trailing characters.

        Parameters
        ----------
        characters:
            The characters to remove. If None, whitespace is removed.

        Returns
        -------
        cell:
            The stripped string.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", ["  ab  ", "~ bc ~", None])
        >>> column.transform(lambda cell: cell.str.strip())
        +--------+
        | a      |
        | ---    |
        | str    |
        +========+
        | ab     |
        | ~ bc ~ |
        | null   |
        +--------+

        >>> column.transform(lambda cell: cell.str.strip(characters=" ~"))
        +------+
        | a    |
        | ---  |
        | str  |
        +======+
        | ab   |
        | bc   |
        | null |
        +------+
        """

    @abstractmethod
    def strip_end(self, *, characters: _ConvertibleToStringCell = None) -> Cell[str | None]:
        """
        Remove trailing characters.

        Parameters
        ----------
        characters:
            The characters to remove. If None, whitespace is removed.

        Returns
        -------
        cell:
            The stripped string.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", ["  ab  ", "~ bc ~", None])
        >>> column.transform(lambda cell: cell.str.strip_end())
        +--------+
        | a      |
        | ---    |
        | str    |
        +========+
        |   ab   |
        | ~ bc ~ |
        | null   |
        +--------+

        >>> column.transform(lambda cell: cell.str.strip_end(characters=" ~"))
        +------+
        | a    |
        | ---  |
        | str  |
        +======+
        |   ab |
        | ~ bc |
        | null |
        +------+
        """

    @abstractmethod
    def strip_start(self, *, characters: _ConvertibleToStringCell = None) -> Cell[str | None]:
        """
        Remove leading characters.

        Parameters
        ----------
        characters:
            The characters to remove. If None, whitespace is removed.

        Returns
        -------
        cell:
            The stripped string.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", ["  ab  ", "~ bc ~", None])
        >>> column.transform(lambda cell: cell.str.strip_start())
        +--------+
        | a      |
        | ---    |
        | str    |
        +========+
        | ab     |
        | ~ bc ~ |
        | null   |
        +--------+

        >>> column.transform(lambda cell: cell.str.strip_start(characters=" ~"))
        +------+
        | a    |
        | ---  |
        | str  |
        +======+
        | ab   |
        | bc ~ |
        | null |
        +------+
        """

    @abstractmethod
    def to_float(self) -> Cell[float | None]:
        """
        Convert the string to a float.

        Returns
        -------
        cell:
            The float value. If the string cannot be converted to a float, None is returned.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", ["1", "1.5", "abc", None])
        >>> column.transform(lambda cell: cell.str.to_float())
        +---------+
        |       a |
        |     --- |
        |     f64 |
        +=========+
        | 1.00000 |
        | 1.50000 |
        |    null |
        |    null |
        +---------+
        """

    @abstractmethod
    def to_int(self, *, base: _ConvertibleToIntCell = 10) -> Cell[int | None]:
        """
        Convert the string to an integer.

        Parameters
        ----------
        base:
            The base of the integer.

        Returns
        -------
        cell:
            The integer value. If the string cannot be converted to an integer, None is returned.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("a", ["1", "10", "abc", None])
        >>> column1.transform(lambda cell: cell.str.to_int())
        +------+
        |    a |
        |  --- |
        |  i64 |
        +======+
        |    1 |
        |   10 |
        | null |
        | null |
        +------+

        >>> column2 = Column("a", ["1", "10", "abc", None])
        >>> column2.transform(lambda cell: cell.str.to_int(base=2))
        +------+
        |    a |
        |  --- |
        |  i64 |
        +======+
        |    1 |
        |    2 |
        | null |
        | null |
        +------+
        """

    @abstractmethod
    def to_lowercase(self) -> Cell[str | None]:
        """
        Convert the string to lowercase.

        Returns
        -------
        cell:
            The lowercase string.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", ["AB", "BC", None])
        >>> column.transform(lambda cell: cell.str.to_lowercase())
        +------+
        | a    |
        | ---  |
        | str  |
        +======+
        | ab   |
        | bc   |
        | null |
        +------+
        """

    @abstractmethod
    def to_uppercase(self) -> Cell[str | None]:
        """
        Convert the string to uppercase.

        Returns
        -------
        cell:
            The uppercase string.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", ["ab", "bc", None])
        >>> column.transform(lambda cell: cell.str.to_uppercase())
        +------+
        | a    |
        | ---  |
        | str  |
        +======+
        | AB   |
        | BC   |
        | null |
        +------+
        """

    # @abstractmethod
    # def contains(self, substring: _ConvertibleToStringCell) -> Cell[bool | None]:
    #     """
    #     Check if the string value in the cell contains the substring.
    #
    #     Parameters
    #     ----------
    #     substring:
    #         The substring to search for.
    #
    #     Returns
    #     -------
    #     contains:
    #         Whether the string value contains the substring.
    #
    #     Examples
    #     --------
    #     >>> from safeds.data.tabular.containers import Column
    #     >>> column = Column("a", ["ab", "bc", "cd", None])
    #     >>> column.transform(lambda cell: cell.str.contains("b"))
    #     +-------+
    #     | a     |
    #     | ---   |
    #     | bool  |
    #     +=======+
    #     | true  |
    #     | true  |
    #     | false |
    #     | null  |
    #     +-------+
    #     """

    # @abstractmethod
    # def index_of(self, substring: _ConvertibleToStringCell) -> Cell[int | None]:
    #     """
    #     Get the index of the first occurrence of the substring in the string value in the cell.
    #
    #     Parameters
    #     ----------
    #     substring:
    #         The substring to search for.
    #
    #     Returns
    #     -------
    #     index_of:
    #         The index of the first occurrence of the substring. If the substring is not found, None is returned.
    #
    #     Examples
    #     --------
    #     >>> from safeds.data.tabular.containers import Column
    #     >>> column = Column("a", ["ab", "bc", "cd", None])
    #     >>> column.transform(lambda cell: cell.str.index_of("b"))
    #     +------+
    #     |    a |
    #     |  --- |
    #     |  u32 |
    #     +======+
    #     |    1 |
    #     |    0 |
    #     | null |
    #     | null |
    #     +------+
    #     """
    #

    # @abstractmethod
    # def replace(self, old: _ConvertibleToStringCell, new: _ConvertibleToStringCell) -> Cell[str | None]:
    #     """
    #     Replace occurrences of the old substring with the new substring in the string value in the cell.
    #
    #     Parameters
    #     ----------
    #     old:
    #         The substring to replace.
    #     new:
    #         The substring to replace with.
    #
    #     Returns
    #     -------
    #     replaced_string:
    #         The string value with the occurrences replaced.
    #
    #     Examples
    #     --------
    #     >>> from safeds.data.tabular.containers import Column
    #     >>> column = Column("a", ["ab", "bc", "cd", None])
    #     >>> column.transform(lambda cell: cell.str.replace("b", "z"))
    #     +------+
    #     | a    |
    #     | ---  |
    #     | str  |
    #     +======+
    #     | az   |
    #     | zc   |
    #     | cd   |
    #     | null |
    #     +------+
    #     """

    # @abstractmethod
    # def substring(
    #     self,
    #     *,
    #     start: _ConvertibleToIntCell = 0,
    #     length: _ConvertibleToIntCell = None,
    # ) -> Cell[str | None]:
    #     """
    #     Get a substring of the string value in the cell.
    #
    #     Parameters
    #     ----------
    #     start:
    #         The start index of the substring.
    #     length:
    #         The length of the substring. If None, the slice contains all rows starting from `start`. Must greater than
    #         or equal to 0.
    #
    #     Returns
    #     -------
    #     substring:
    #         The substring of the string value.
    #
    #     Raises
    #     ------
    #     OutOfBoundsError
    #         If length is less than 0.
    #
    #     Examples
    #     --------
    #     >>> from safeds.data.tabular.containers import Column
    #     >>> column = Column("a", ["abc", "def", "ghi", None])
    #     >>> column.transform(lambda cell: cell.str.substring(start=1, length=2))
    #     +------+
    #     | a    |
    #     | ---  |
    #     | str  |
    #     +======+
    #     | bc   |
    #     | ef   |
    #     | hi   |
    #     | null |
    #     +------+
    #     """
    #
    # # TODO: add format parameter + document
    # @abstractmethod
    # def to_date(self, *, format: str | None = "iso") -> Cell[datetime.date | None]:
    #     """
    #     Convert the string value in the cell to a date.
    #
    #     Returns
    #     -------
    #     date:
    #         The date value. If the string cannot be converted to a date, None is returned.
    #
    #     Examples
    #     --------
    #     >>> from safeds.data.tabular.containers import Column
    #     >>> column = Column("a", ["2021-01-01", "2021-02-01", "abc", None])
    #     >>> column.transform(lambda cell: cell.str.to_date())
    #     +------------+
    #     | a          |
    #     | ---        |
    #     | date       |
    #     +============+
    #     | 2021-01-01 |
    #     | 2021-02-01 |
    #     | null       |
    #     | null       |
    #     +------------+
    #     """
    #
    # # TODO: add format parameter + document
    # @abstractmethod
    # def to_datetime(self, *, format: str | None = "iso") -> Cell[datetime.datetime | None]:
    #     """
    #     Convert the string value in the cell to a datetime.
    #
    #     Returns
    #     -------
    #     datetime:
    #         The datetime value. If the string cannot be converted to a datetime, None is returned.
    #
    #     Examples
    #     --------
    #     >>> from safeds.data.tabular.containers import Column
    #     >>> column = Column("a", ["2021-01-01T00:00:00Z", "2021-02-01T00:00:00Z", "abc", None])
    #     >>> column.transform(lambda cell: cell.str.to_datetime())
    #     +-------------------------+
    #     | a                       |
    #     | ---                     |
    #     | datetime[μs, UTC]       |
    #     +=========================+
    #     | 2021-01-01 00:00:00 UTC |
    #     | 2021-02-01 00:00:00 UTC |
    #     | null                    |
    #     | null                    |
    #     +-------------------------+
    #     """
    #
    # # TODO: add to_time
    #
