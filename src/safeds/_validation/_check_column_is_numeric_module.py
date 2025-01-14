"""The module name must differ from the function name, so it can be re-exported properly with apipkg."""

from __future__ import annotations

from typing import TYPE_CHECKING

from safeds.exceptions import ColumnTypeError

if TYPE_CHECKING:
    from collections.abc import Container

    from safeds.data.tabular.containers import Column, Table
    from safeds.data.tabular.typing import Schema


def _check_column_is_numeric(
    column: Column,
    *,
    other_columns: list[Column] | None = None,
    operation: str = "do a numeric operation",
) -> None:
    """
    Check whether a column is numeric, and raise an error if it is not.

    Parameters
    ----------
    column:
        The column to check.
    other_columns:
        Other columns to check. This provides better error messages than checking each column individually.
    operation:
        The operation that is performed on the column. This is used in the error message.

    Raises
    ------
    ColumnTypeError
        If a column is not numeric.
    """
    if other_columns is None:
        other_columns = []

    columns = [column, *other_columns]
    non_numeric_names = [col.name for col in columns if not col.type.is_numeric]

    if non_numeric_names:
        message = _build_error_message(non_numeric_names, operation)
        raise ColumnTypeError(message) from None


def _check_columns_are_numeric(
    table_or_schema: Table | Schema,
    selector: str | list[str],
    *,
    operation: str = "do a numeric operation",
) -> None:
    """
    Check if the specified columns are numeric and raise an error if they are not. Missing columns are ignored.

    Parameters
    ----------
    table_or_schema:
        The table or schema to check.
    selector:
        The columns to check.
    operation:
        The operation that is performed on the columns. This is used in the error message.

    Raises
    ------
    ColumnTypeError
        If a column exists but is not numeric.
    """
    from safeds.data.tabular.containers import Table  # circular import

    if isinstance(table_or_schema, Table):
        table_or_schema = table_or_schema.schema
    if isinstance(selector, str):  # pragma: no cover
        selector = [selector]

    if len(selector) > 1:
        # Create a set for faster containment checks
        known_names: Container = set(table_or_schema.column_names)
    else:
        known_names = table_or_schema.column_names

    non_numeric_names = [
        name for name in selector if name in known_names and not table_or_schema.get_column_type(name).is_numeric
    ]
    if non_numeric_names:
        message = _build_error_message(non_numeric_names, operation)
        raise ColumnTypeError(message) from None


def _build_error_message(non_numeric_names: list[str], operation: str) -> str:
    return f"Tried to {operation} on non-numeric columns {non_numeric_names}."
