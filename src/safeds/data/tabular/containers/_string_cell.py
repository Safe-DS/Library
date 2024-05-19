from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from safeds.data.tabular.containers import Cell


class StringCell(ABC):
    """
    Namespace for operations on strings.

    This class cannot be instantiated directly. It can only be accessed using the `string` attribute of a cell.

    Examples
    --------
    >>> from safeds.data.tabular.containers import Column
    >>> column = Column("example", ["ab", "bc", "cd"])
    >>> column.transform(lambda cell: cell.string.to_uppercase())
    ["AB", "BC", "CD"]
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
        >>> column.count_if(lambda cell: cell.string.contains("b"))
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
        >>> column.count_if(lambda cell: cell.string.ends_with("c"))
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
        >>> column.transform(lambda cell: cell.string.index_of("b"))
        [1, 0, None]
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
        >>> column.transform(lambda cell: cell.string.length())
        [0, 1, 3]
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
        >>> column.count_if(lambda cell: cell.string.starts_with("a"))
        1
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
        >>> column.transform(lambda cell: cell.string.to_float())
        [1.0, 3.4, 5.6, None]
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
        >>> column1.transform(lambda cell: cell.string.to_int())
        [1, 2, 3, None]

        >>> column2 = Column("example", ["1", "10", "11", "abc"])
        >>> column2.transform(lambda cell: cell.string.to_int(base=2))
        [1, 2, 3, None]
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
        >>> column.transform(lambda cell: cell.string.to_lowercase())
        ["ab", "bc", "cd"]
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
        >>> column.transform(lambda cell: cell.string.to_uppercase())
        ["AB", "BC", "CD"]
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
        >>> column.transform(lambda cell: cell.string.trim())
        ["", "abc", "abc", "abc"]
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
        >>> column.transform(lambda cell: cell.string.trim_end())
        ["", " abc", "abc", " abc"]
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
        >>> column.transform(lambda cell: cell.string.trim_start())
        ["", "abc", "abc ", "abc "]
        """

    # replace -> replace/replace_many/replace_all
    # substring -> slice
    # toDate -> to_date
    # toTime -> to_time
    # toDatetime -> to_datetime
