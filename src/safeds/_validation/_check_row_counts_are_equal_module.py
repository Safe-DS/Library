"""The module name must differ from the function name, so it can be re-exported properly with apipkg."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from safeds.exceptions import LengthMismatchError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from safeds.data.tabular.containers import Column, Table


def _check_row_counts_are_equal(data: Sequence[Column | Table] | Mapping[str, Sequence[Any]]) -> None:
    """
    Check whether all columns or tables have the same row count, and raise an error if they do not.

    Parameters
    ----------
    data:
        The columns or tables to check.

    Raises
    ------
    LengthMismatchError
        If some columns or tables have different row counts.
    """
    if len(data) < 2:
        return

    # Compute the mismatched columns
    names_and_row_counts = _get_names_and_row_counts(data)
    first_name, first_row_count = names_and_row_counts[0]
    mismatched_columns: list[tuple[str, int]] = []

    for entry in names_and_row_counts[1:]:
        if entry[1] != first_row_count:
            mismatched_columns.append(entry)

    # Raise an error if there are mismatched columns
    if mismatched_columns:
        message = _build_error_message(names_and_row_counts[0], mismatched_columns)
        raise LengthMismatchError(message)


def _get_names_and_row_counts(data: Sequence[Column | Table] | Mapping[str, Sequence[Any]]) -> list[tuple[str, int]]:
    from safeds.data.tabular.containers import Column, Table  # circular import

    if isinstance(data, Mapping):
        return [(name, len(column)) for name, column in data.items()]
    else:
        result = []

        for i, entry in enumerate(data):
            if isinstance(entry, Column):
                result.append((entry.name, entry.row_count))
            elif isinstance(entry, Table):
                result.append((f"table_{i}", entry.row_count))

        return result


def _build_error_message(first_entry: tuple[str, int], mismatched_entries: list[tuple[str, int]]) -> str:
    result = f"'{first_entry[0]}' has {first_entry[1]} rows, which differs from:"

    for entry in mismatched_entries:
        result += f"\n    - '{entry[0]}' ({entry[1]} rows)"

    return result
