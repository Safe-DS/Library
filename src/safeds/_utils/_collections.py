from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def _compute_duplicates(values: list[T], *, forbidden_values: set[T] | None = None) -> list[T]:
    """
    Compute the duplicates in a list of values.

    Parameters
    ----------
    values:
        The values to check for duplicates.
    forbidden_values:
        Additional values that are considered duplicates if they occur. Defaults to an empty set.

    Returns
    -------
    duplicates:
        The duplicates in the list of values.
    """
    if forbidden_values is None:
        forbidden_values = set()

    duplicates = []
    for value in values:
        if value in forbidden_values:
            duplicates.append(value)
        else:
            forbidden_values.add(value)

    return duplicates
