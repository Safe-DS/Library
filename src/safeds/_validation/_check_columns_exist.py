from __future__ import annotations

from typing import TYPE_CHECKING

from safeds.exceptions import ColumnNotFoundError

if TYPE_CHECKING:
    from collections.abc import Container

    from safeds.data.tabular.containers import Table
    from safeds.data.tabular.typing import Schema


def _check_columns_exist(table_or_schema: Table | Schema, requested_names: str | list[str]) -> None:
    """
    Check if the specified column names exist in the table and raise an error if they do not.

    Parameters
    ----------
    table_or_schema:
        The table or schema to check.
    requested_names:
        The column names to check.

    Raises
    ------
    KeyError
        If a column name does not exist.
    """
    from safeds.data.tabular.containers import Table  # circular import

    if isinstance(table_or_schema, Table):
        table_or_schema = table_or_schema.schema
    if isinstance(requested_names, str):
        requested_names = [requested_names]

    if len(requested_names) > 1:
        known_names: Container = set(table_or_schema.column_names)
    else:
        known_names = table_or_schema.column_names

    unknown_names = [name for name in requested_names if name not in known_names]
    if unknown_names:
        raise ColumnNotFoundError(unknown_names)  # TODO: in the error, compute similar column names


def _get_similar_column_names(schema: Schema, column_name: str) -> list[str]:
    """
    Get similar column names to the specified column name.

    Parameters
    ----------
    schema:
        The schema to check.
    column_name:
        The column name to check.

    Returns
    -------
    similar_columns:
        The similar column names.
    """
    from difflib import get_close_matches

    return get_close_matches(
        column_name,
        schema.column_names,
        n=3,
    )
