from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, TypeVar

from safeds.exceptions import EmptyChoiceError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

T = TypeVar("T")


class Choice(Collection[T]):
    """A list of values to choose from in a hyperparameter search."""

    def __init__(self, *args: T) -> None:
        """
        Create a new choice. Duplicate values will be removed.

        Duplicate values will be removed.

        Parameters
        ----------
        *args:
            The values to choose from.
        """
        if len(args) < 1:
            raise EmptyChoiceError
        self.elements = list(dict.fromkeys(args))

    def __contains__(self, value: Any) -> bool:
        """
        Check if a value is in this choice.

        Parameters
        ----------
        value:
            The value to check.

        Returns
        -------
        is_in_choice:
            Whether the value is in this choice.
        """
        return value in self.elements

    def __iter__(self) -> Iterator[T]:
        """
        Iterate over the values of this choice.

        Returns
        -------
        iterator:
            An iterator over the values of this choice.
        """
        return iter(self.elements)

    def __len__(self) -> int:
        """
        Get the number of values in this choice.

        Returns
        -------
        value_count:
            The number of values in this choice.
        """
        return len(self.elements)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Choice):
            return NotImplemented
        if self is other:
            return True
        return (self is other) or (self.elements == other.elements)
