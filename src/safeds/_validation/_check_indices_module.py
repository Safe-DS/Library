"""The module name must differ from the function name, so it can be re-exported properly with apipkg."""

from __future__ import annotations

from typing import TYPE_CHECKING

from safeds.exceptions import IndexOutOfBoundsError

if TYPE_CHECKING:
    from collections.abc import Sequence


def _check_indices(
    sequence: Sequence,
    indices: int | list[int],
    *,
    allow_negative: bool = True,
) -> None:
    """
    Check if indices are valid for the provided sequence.

    Parameters
    ----------
    sequence:
        The sequence to check.
    indices:
        The indices to check.
    allow_negative:
        If negative indices are allowed.

    Raises
    ------
    IndexOutOfBoundsError:
        If the index is out of bounds.
    """
    if isinstance(indices, int):
        indices = [indices]

    min_legal = -len(sequence) if allow_negative else 0
    max_legal = len(sequence) - 1

    illegal_indices = [index for index in indices if not min_legal <= index <= max_legal]
    if illegal_indices:
        message = _build_error_message(illegal_indices, min_legal, max_legal)
        raise IndexOutOfBoundsError(message) from None


def _build_error_message(illegal_indices: list[int], min_legal: int, max_legal: int) -> str:
    return f"The indices {illegal_indices} are outside the legal interval [{min_legal}, {max_legal}]."
