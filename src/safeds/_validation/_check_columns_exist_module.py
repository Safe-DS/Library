"""The module name must differ from the function name, so it can be re-exported properly with apipkg."""

from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _get_similar_strings
from safeds.exceptions import ColumnNotFoundError

if TYPE_CHECKING:
    from collections.abc import Container

    from safeds.data.tabular.containers import Table
    from safeds.data.tabular.typing import Schema


def _check_columns_exist(table_or_schema: Table | Schema, selector: str | list[str]) -> None:
    """
    Check whether the specified columns exist, and raise an error if they do not.

    Parameters
    ----------
    table_or_schema:
        The table or schema to check.
    selector:
        The columns to check.

    Raises
    ------
    ColumnNotFoundError
        If a column name does not exist.
    """
    from safeds.data.tabular.containers import Table  # circular import

    if isinstance(table_or_schema, Table):
        table_or_schema = table_or_schema.schema
    if isinstance(selector, str):
        selector = [selector]

    if len(selector) > 1:
        # Create a set for faster containment checks
        known_names: Container = set(table_or_schema.column_names)
    else:
        known_names = table_or_schema.column_names

    unknown_names = [name for name in selector if name not in known_names]
    if unknown_names:
        message = _build_error_message(table_or_schema, unknown_names)
        raise ColumnNotFoundError(message) from None


def _build_error_message(schema: Schema, unknown_names: list[str]) -> str:
    result = "Could not find column(s):"

    for unknown_name in unknown_names:
        similar_columns = _get_similar_strings(unknown_name, schema.column_names)
        result += f"\n    - '{unknown_name}'"
        if similar_columns:
            result += f": Did you mean one of {similar_columns}?"

    return result
