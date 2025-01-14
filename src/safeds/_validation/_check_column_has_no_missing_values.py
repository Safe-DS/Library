"""The module name must differ from the function name, so it can be re-exported properly with apipkg."""

from __future__ import annotations

from typing import TYPE_CHECKING

from safeds.exceptions import MissingValuesError

if TYPE_CHECKING:
    from safeds.data.tabular.containers import Column


def _check_column_has_no_missing_values(
    column: Column,
    *,
    other_columns: list[Column] | None = None,
    operation: str = "do an operation",
) -> None:
    """
    Check if the column has no missing values.

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
    MissingValuesError:
        If a has missing values.
    """
    if other_columns is None:  # pragma: no cover
        other_columns = []

    columns = [column, *other_columns]
    missing_values_columns = [column.name for column in columns if column._series.has_nulls()]

    if missing_values_columns:
        message = _build_error_message(missing_values_columns, operation)
        raise MissingValuesError(message) from None


def _build_error_message(missing_values_names: list[str], operation: str) -> str:
    return f"Tried to {operation} on columns with missing values {missing_values_names}."
