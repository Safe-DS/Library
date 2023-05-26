from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

T = TypeVar("T")


class Choice(Collection[T]):
    """A list of values to choose from in a hyperparameter search."""

    def __init__(self, *args: T) -> None:
        """
        Create a new choice.

        Parameters
        ----------
        *args: tuple[T, ...]
            The values to choose from.
        """
        self.elements = list(args)

    def __contains__(self, value: Any) -> bool:
        """
        Check if a value is in this choice.

        Parameters
        ----------
        value: Any
            The value to check.

        Returns
        -------
        is_in_choice : bool
            Whether the value is in this choice.
        """
        return value in self.elements

    def __iter__(self) -> Iterator[T]:
        """
        Iterate over the values of this choice.

        Returns
        -------
        iterator : Iterator[T]
            An iterator over the values of this choice.
        """
        return iter(self.elements)

    def __len__(self) -> int:
        """
        Get the number of values in this choice.

        Returns
        -------
        number_of_values : int
            The number of values in this choice.
        """
        return len(self.elements)
