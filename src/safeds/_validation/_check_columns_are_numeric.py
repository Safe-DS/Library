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
    operation: str = "do a numeric operation",
) -> None:
    """
    Check if the column is numeric and raise an error if it is not.

    Parameters
    ----------
    column:
        The column to check.
    operation:
        The operation that is performed on the column. This is used in the error message.

    Raises
    ------
    ColumnTypeError
        If the column is not numeric.
    """
    if not column.type.is_numeric:
        message = _build_error_message([column.name], operation)
        raise ColumnTypeError(message)


def _check_columns_are_numeric(
    table_or_schema: Table | Schema,
    column_names: str | list[str],
    *,
    operation: str = "do a numeric operation",
) -> None:
    """
    Check if the columns with the specified names are numeric and raise an error if they are not.

    Missing columns are ignored. Use `_check_columns_exist` to check for missing columns.

    Parameters
    ----------
    table_or_schema:
        The table or schema to check.
    column_names:
        The column names to check.
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
    if isinstance(column_names, str):
        column_names = [column_names]

    if len(column_names) > 1:
        # Create a set for faster containment checks
        known_names: Container = set(table_or_schema.column_names)
    else:
        known_names = table_or_schema.column_names

    non_numeric_names = [
        name for name in column_names if name in known_names and not table_or_schema.get_column_type(name).is_numeric
    ]
    if non_numeric_names:
        message = _build_error_message(non_numeric_names, operation)
        raise ColumnTypeError(message)


def _build_error_message(non_numeric_names: list[str], operation: str) -> str:
    return f"Tried to {operation} on non-numeric columns {non_numeric_names}."
