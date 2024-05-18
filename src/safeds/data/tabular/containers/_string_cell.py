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
