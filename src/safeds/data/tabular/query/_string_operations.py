from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import datetime

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
    def contains(self, substring: _ConvertibleToStringCell) -> Cell[bool | None]:
        """
        Check if the string contains the substring.

        Parameters
        ----------
        substring:
            The substring to search for.

        Returns
        -------
        contains:
            Whether the string contains the substring.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", ["ab", "cd", None])
        >>> column.transform(lambda cell: cell.str.contains("b"))
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
    def index_of(self, substring: _ConvertibleToStringCell) -> Cell[int | None]:
        """
        Get the index of the first occurrence of the substring.

        Parameters
        ----------
        substring:
            The substring to search for.

        Returns
        -------
        cell:
            The index of the first occurrence of the substring. If the substring is not found, None is returned.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", ["ab", "cd", None])
        >>> column.transform(lambda cell: cell.str.index_of("b"))
        +------+
        |    a |
        |  --- |
        |  u32 |
        +======+
        |    1 |
        | null |
        | null |
        +------+
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
    def remove_prefix(self, prefix: _ConvertibleToStringCell) -> Cell[str | None]:
        """
        Remove a prefix from the string. Strings without the prefix are not changed.

        Parameters
        ----------
        prefix:
            The prefix to remove.

        Returns
        -------
        cell:
            The string without the prefix.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", ["ab", "bc", None])
        >>> column.transform(lambda cell: cell.str.remove_prefix("a"))
        +------+
        | a    |
        | ---  |
        | str  |
        +======+
        | b    |
        | bc   |
        | null |
        +------+
        """

    @abstractmethod
    def remove_suffix(self, suffix: _ConvertibleToStringCell) -> Cell[str | None]:
        """
        Remove a suffix from the string. Strings without the suffix are not changed.

        Parameters
        ----------
        suffix:
            The suffix to remove.

        Returns
        -------
        cell:
            The string without the suffix.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", ["ab", "bc", None])
        >>> column.transform(lambda cell: cell.str.remove_suffix("b"))
        +------+
        | a    |
        | ---  |
        | str  |
        +======+
        | a    |
        | bc   |
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
    def replace_all(self, old: _ConvertibleToStringCell, new: _ConvertibleToStringCell) -> Cell[str | None]:
        """
        Replace all occurrences of the old substring with the new substring.

        Parameters
        ----------
        old:
            The substring to replace.
        new:
            The substring to replace with.

        Returns
        -------
        cell:
            The string with all occurrences replaced.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", ["ab", "bc", None])
        >>> column.transform(lambda cell: cell.str.replace_all("b", "z"))
        +------+
        | a    |
        | ---  |
        | str  |
        +======+
        | az   |
        | zc   |
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
    def to_date(self, *, format: str | None = "iso") -> Cell[datetime.date | None]:
        r"""
        Convert a string to a date.

        The `format` parameter controls the presentation. It can be `"iso"` to target ISO 8601 or a custom string. The
        custom string can contain fixed specifiers (see below), which are replaced with the corresponding values. The
        specifiers are case-sensitive and always enclosed in curly braces. Other text is included in the output
        verbatim. To include a literal opening curly brace, use `\{`, and to include a literal backslash, use `\\`.

        The following specifiers are available:

        - `{Y}`, `{_Y}`, `{^Y}`: Year (zero-padded to four digits, space-padded to four digits, no padding).
        - `{Y99}`, `{_Y99}`, `{^Y99}`: Year modulo 100 (zero-padded to two digits, space-padded to two digits, no
          padding).
        - `{M}`, `{_M}`, `{^M}`: Month (zero-padded to two digits, space-padded to two digits, no padding).
        - `{M-full}`: Full name of the month (e.g. "January").
        - `{M-short}`: Abbreviated name of the month with three letters (e.g. "Jan").
        - `{W}`, `{_W}`, `{^W}`: Week number as defined by ISO 8601 (zero-padded to two digits, space-padded to two
          digits, no padding).
        - `{D}`, `{_D}`, `{^D}`: Day of the month (zero-padded to two digits, space-padded to two digits, no padding).
        - `{DOW}`: Day of the week as defined by ISO 8601 (1 = Monday, 7 = Sunday).
        - `{DOW-full}`: Full name of the day of the week (e.g. "Monday").
        - `{DOW-short}`: Abbreviated name of the day of the week with three letters (e.g. "Mon").
        - `{DOY}`, `{_DOY}`, `{^DOY}`: Day of the year, ranging from 1 to 366 (zero-padded to three digits, space-padded
          to three digits, no padding).

        The specifiers follow certain conventions:

        - If a component may be formatted in multiple ways, we use shorter specifiers for ISO 8601. Specifiers for
          other formats have a prefix (same value with different padding, see below) or suffix (other differences).
        - By default, value are zero-padded, where applicable.
        - A leading underscore (`_`) means the value is space-padded.
        - A leading caret (`^`) means the value has no padding (think of the caret in regular expressions).

        Parameters
        ----------
        format:
            The format to use.

        Returns
        -------
        cell:
            The parsed date.

        Raises
        ------
        ValueError
            If the format is invalid.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", ["1999-02-03", "03.02.2001", "abc", None])
        >>> column.transform(lambda cell: cell.str.to_date())
        +------------+
        | a          |
        | ---        |
        | date       |
        +============+
        | 1999-02-03 |
        | null       |
        | null       |
        | null       |
        +------------+

        >>> column.transform(lambda cell: cell.str.to_date(format="{D}.{M}.{Y}"))
        +------------+
        | a          |
        | ---        |
        | date       |
        +============+
        | null       |
        | 2001-02-03 |
        | null       |
        | null       |
        +------------+
        """

    @abstractmethod
    def to_datetime(self, *, format: str | None = "iso") -> Cell[datetime.datetime | None]:
        r"""
        Convert a string to a datetime.

        The `format` parameter controls the presentation. It can be `"iso"` to target ISO 8601 or a custom string. The
        custom string can contain fixed specifiers (see below), which are replaced with the corresponding values. The
        specifiers are case-sensitive and always enclosed in curly braces. Other text is included in the output
        verbatim. To include a literal opening curly brace, use `\{`, and to include a literal backslash, use `\\`.

        The following specifiers for _date components_ are available for **datetime** and **date**:

        - `{Y}`, `{_Y}`, `{^Y}`: Year (zero-padded to four digits, space-padded to four digits, no padding).
        - `{Y99}`, `{_Y99}`, `{^Y99}`: Year modulo 100 (zero-padded to two digits, space-padded to two digits, no
          padding).
        - `{M}`, `{_M}`, `{^M}`: Month (zero-padded to two digits, space-padded to two digits, no padding).
        - `{M-full}`: Full name of the month (e.g. "January").
        - `{M-short}`: Abbreviated name of the month with three letters (e.g. "Jan").
        - `{W}`, `{_W}`, `{^W}`: Week number as defined by ISO 8601 (zero-padded to two digits, space-padded to two
          digits, no padding).
        - `{D}`, `{_D}`, `{^D}`: Day of the month (zero-padded to two digits, space-padded to two digits, no padding).
        - `{DOW}`: Day of the week as defined by ISO 8601 (1 = Monday, 7 = Sunday).
        - `{DOW-full}`: Full name of the day of the week (e.g. "Monday").
        - `{DOW-short}`: Abbreviated name of the day of the week with three letters (e.g. "Mon").
        - `{DOY}`, `{_DOY}`, `{^DOY}`: Day of the year, ranging from 1 to 366 (zero-padded to three digits, space-padded
          to three digits, no padding).

        The following specifiers for _time components_ are available for **datetime** and **time**:

        - `{h}`, `{_h}`, `{^h}`: Hour (zero-padded to two digits, space-padded to two digits, no padding).
        - `{h12}`, `{_h12}`, `{^h12}`: Hour in 12-hour format (zero-padded to two digits, space-padded to two digits, no
          padding).
        - `{m}`, `{_m}`, `{^m}`: Minute (zero-padded to two digits, space-padded to two digits, no padding).
        - `{s}`, `{_s}`, `{^s}`: Second (zero-padded to two digits, space-padded to two digits, no padding).
        - `{.f}`: Fractional seconds with a leading decimal point.
        - `{ms}`: Millisecond (zero-padded to three digits).
        - `{us}`: Microsecond (zero-padded to six digits).
        - `{ns}`: Nanosecond (zero-padded to nine digits).
        - `{AM/PM}`: AM or PM (uppercase).
        - `{am/pm}`: am or pm (lowercase).

        The following specifiers are available for **datetime** only:

        - `{z}`: Offset of the timezone from UTC without a colon (e.g. "+0000").
        - `{:z}`: Offset of the timezone from UTC with a colon (e.g. "+00:00").
        - `{u}`: The UNIX timestamp in seconds.

        The specifiers follow certain conventions:

        - Generally, date components use uppercase letters and time components use lowercase letters.
        - If a component may be formatted in multiple ways, we use shorter specifiers for ISO 8601. Specifiers for
          other formats have a prefix (same value with different padding, see below) or suffix (other differences).
        - By default, value are zero-padded, where applicable.
        - A leading underscore (`_`) means the value is space-padded.
        - A leading caret (`^`) means the value has no padding (think of the caret in regular expressions).

        Parameters
        ----------
        format:
            The format to use.

        Returns
        -------
        cell:
            The parsed datetime.

        Raises
        ------
        ValueError
            If the format is invalid.

        Examples
        --------
        >>> from datetime import date, datetime
        >>> from safeds.data.tabular.containers import Column
        >>> column1 = Column("a", ["1999-12-31T01:02:03Z", "12:30 Jan 23 2024", "abc", None])
        >>> column1.transform(lambda cell: cell.str.to_datetime())
        +-------------------------+
        | a                       |
        | ---                     |
        | datetime[μs, UTC]       |
        +=========================+
        | 1999-12-31 01:02:03 UTC |
        | null                    |
        | null                    |
        | null                    |
        +-------------------------+

        >>> column1.transform(lambda cell: cell.str.to_datetime(
        ...     format="{h}:{m} {M-short} {D} {Y}"
        ... ))
        +---------------------+
        | a                   |
        | ---                 |
        | datetime[μs]        |
        +=====================+
        | null                |
        | 2024-01-23 12:30:00 |
        | null                |
        | null                |
        +---------------------+
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
    def to_time(self, *, format: str | None = "iso") -> Cell[datetime.time | None]:
        r"""
        Convert a string to a time.

        The `format` parameter controls the presentation. It can be `"iso"` to target ISO 8601 or a custom string. The
        custom string can contain fixed specifiers (see below), which are replaced with the corresponding values. The
        specifiers are case-sensitive and always enclosed in curly braces. Other text is included in the output
        verbatim. To include a literal opening curly brace, use `\{`, and to include a literal backslash, use `\\`.

        The following specifiers are available:

        - `{h}`, `{_h}`, `{^h}`: Hour (zero-padded to two digits, space-padded to two digits, no padding).
        - `{h12}`, `{_h12}`, `{^h12}`: Hour in 12-hour format (zero-padded to two digits, space-padded to two digits, no
          padding).
        - `{m}`, `{_m}`, `{^m}`: Minute (zero-padded to two digits, space-padded to two digits, no padding).
        - `{s}`, `{_s}`, `{^s}`: Second (zero-padded to two digits, space-padded to two digits, no padding).
        - `{.f}`: Fractional seconds with a leading decimal point.
        - `{ms}`: Millisecond (zero-padded to three digits).
        - `{us}`: Microsecond (zero-padded to six digits).
        - `{ns}`: Nanosecond (zero-padded to nine digits).
        - `{AM/PM}`: AM or PM (uppercase).
        - `{am/pm}`: am or pm (lowercase).

        The specifiers follow certain conventions:

        - If a component may be formatted in multiple ways, we use shorter specifiers for ISO 8601. Specifiers for
          other formats have a prefix (same value with different padding, see below) or suffix (other differences).
        - By default, value are zero-padded, where applicable.
        - A leading underscore (`_`) means the value is space-padded.
        - A leading caret (`^`) means the value has no padding (think of the caret in regular expressions).

        Parameters
        ----------
        format:
            The format to use.

        Returns
        -------
        cell:
            The parsed time.

        Raises
        ------
        ValueError
            If the format is invalid.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("a", ["12:34", "12:34:56", "12:34:56.789", "abc", None])
        >>> column.transform(lambda cell: cell.str.to_time())
        +--------------+
        | a            |
        | ---          |
        | time         |
        +==============+
        | null         |
        | 12:34:56     |
        | 12:34:56.789 |
        | null         |
        | null         |
        +--------------+

        >>> column.transform(lambda cell: cell.str.to_time(format="{h}:{m}"))
        +----------+
        | a        |
        | ---      |
        | time     |
        +==========+
        | 12:34:00 |
        | null     |
        | null     |
        | null     |
        | null     |
        +----------+
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
