from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from safeds.data.tabular.containers import Cell


class StringCell(ABC):
    """These operations only make sense for string cells."""

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
        result:
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
        result:
            Whether the string value ends with the suffix.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", ["ab", "bc", "cd"])
        >>> column.count_if(lambda cell: cell.string.ends_with("c"))
        1
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
        result:
            Whether the string value starts with the prefix.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Column
        >>> column = Column("example", ["ab", "bc", "cd"])
        >>> column.count_if(lambda cell: cell.string.starts_with("a"))
        1
        """


    # indexOf
    # lastIndexOf
    # replace
    # split
    # substring
    # toFloat
    # toInt
    # toLowercase
    # toUppercase
    # trim
    # trimEnd
    # trimStart
    # toDate
    # toTime
    # toDatetime
