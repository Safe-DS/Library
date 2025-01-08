from __future__ import annotations

from typing import TYPE_CHECKING

from safeds.exceptions import DuplicateColumnError

if TYPE_CHECKING:
    from safeds.data.tabular.containers import Table
    from safeds.data.tabular.typing import Schema


def _check_columns_dont_exist(
    table_or_schema: Table | Schema,
    new_names: str | list[str],
    *,
    old_name: str | None = None,
) -> None:
    """
    Check whether the specified new column names do not exist yet and are unique, and raise an error if they do.

    Parameters
    ----------
    table_or_schema:
        The table or schema to check.
    new_names:
        The column names to check.
    old_name:
        The old column name to exclude from the check. Set this to None if you don't want to exclude any column.

    Raises
    ------
    DuplicateColumnError
        If a column name exists already.
    """
    from safeds.data.tabular.containers import Table  # circular import

    if isinstance(table_or_schema, Table):
        table_or_schema = table_or_schema.schema
    if isinstance(new_names, str):
        new_names = [new_names]

    # Compute the duplicate names
    known_names = set(table_or_schema.column_names) - {old_name}
    duplicate_names = []

    for name in new_names:
        if name in known_names:
            duplicate_names.append(name)
        else:
            known_names.add(name)

    # Raise an error if duplicate names exist
    if duplicate_names:
        message = _build_error_message(duplicate_names)
        raise DuplicateColumnError(message)


def _build_error_message(duplicate_names: list[str]) -> str:
    return f"The columns {duplicate_names} exist already."
