from __future__ import annotations

from typing import TYPE_CHECKING

from safeds.exceptions import ColumnNotFoundError

if TYPE_CHECKING:
    from collections.abc import Container

    from safeds.data.tabular.containers import Table
    from safeds.data.tabular.typing import Schema


def _check_columns_exist(table_or_schema: Table | Schema, requested_names: str | list[str]) -> None:
    """
    Check if the specified column names exist in the table or schema and raise an error if they do not.

    Parameters
    ----------
    table_or_schema:
        The table or schema to check.
    requested_names:
        The column names to check.

    Raises
    ------
    ColumnNotFoundError
        If a column name does not exist.
    """
    from safeds.data.tabular.containers import Table  # circular import

    if isinstance(table_or_schema, Table):
        table_or_schema = table_or_schema.schema
    if isinstance(requested_names, str):
        requested_names = [requested_names]

    if len(requested_names) > 1:
        # Create a set for faster containment checks
        known_names: Container = set(table_or_schema.column_names)
    else:
        known_names = table_or_schema.column_names

    unknown_names = [name for name in requested_names if name not in known_names]
    if unknown_names:
        message = _build_error_message(table_or_schema, unknown_names)
        raise ColumnNotFoundError(message)


def _build_error_message(schema: Schema, unknown_names: list[str]) -> str:
    message = "Could not find column(s):"

    for unknown_name in unknown_names:
        similar_columns = _get_similar_column_names(schema, unknown_name)
        message += f"\n    - '{unknown_name}'"
        if similar_columns:
            message += f": Did you mean one of {similar_columns}?"

    return message


def _get_similar_column_names(schema: Schema, unknown_name: str) -> list[str]:
    from difflib import get_close_matches

    return get_close_matches(
        unknown_name,
        schema.column_names,
        n=3,
    )
