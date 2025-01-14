"""The module name must differ from the function name, so it can be re-exported properly with apipkg."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

from safeds.exceptions import SchemaError

if TYPE_CHECKING:
    import polars as pl

    from safeds.data.tabular.containers import Table
    from safeds.data.tabular.typing import Schema


_TypeCheckingMode: TypeAlias = Literal["equality", "off"]


def _check_schema(
    expected: Table | Schema,
    actual: Table | Schema,
    *,
    check_names_exactly: bool = True,
    check_types: _TypeCheckingMode = "equality",
) -> None:
    """
    Check whether several schemas match, and raise an error if they do not.

    Parameters
    ----------
    expected:
        The expected schema.
    actual:
        The actual schema.
    check_names_exactly:
        Whether to check that column names of the expected and actual schema are equal (including order). If this is
        turned off, the actual column names can be any superset of the expected column names.
    check_types:
        Whether to check the types of the columns. If "equal", the types must be equal. If "off", the types are not
        checked.

    Raises
    ------
    SchemaError
        If the schemas do not match.
    """
    from safeds.data.tabular.containers import Table  # circular import

    expected_schema: Schema = expected.schema if isinstance(expected, Table) else expected
    actual_schema: Schema = actual.schema if isinstance(actual, Table) else actual

    expected_column_names = expected_schema.column_names
    actual_column_names = actual_schema.column_names

    # All columns must exist
    missing_columns = set(expected_column_names) - set(actual_column_names)
    if missing_columns:
        message = _build_error_message_for_missing_columns(sorted(missing_columns))
        raise SchemaError(message) from None

    # There must be no additional columns
    if check_names_exactly:
        additional_columns = set(actual_column_names) - set(expected_column_names)
        if additional_columns:
            message = _build_error_message_for_additional_columns(sorted(additional_columns))
            raise SchemaError(message) from None

    # All columns must have the correct order
    if check_names_exactly and expected_column_names != actual_column_names:
        message = _build_error_message_for_columns_in_wrong_order(expected_column_names, actual_column_names)
        raise SchemaError(message) from None

    # All columns must have the correct type
    _check_types(expected_schema, actual_schema, check_types=check_types)


def _check_types(expected_schema: Schema, actual_schema: Schema, *, check_types: _TypeCheckingMode) -> None:
    if check_types == "off":  # pragma: no cover
        return

    mismatched_types: list[tuple[str, pl.DataType, pl.DataType]] = []

    for column_name in expected_schema.column_names:
        expected_polars_dtype = _get_polars_dtype(expected_schema, column_name)
        actual_polars_dtype = _get_polars_dtype(actual_schema, column_name)

        if expected_polars_dtype is None or actual_polars_dtype is None:  # pragma: no cover
            continue

        if check_types == "equality" and not actual_polars_dtype.is_(expected_polars_dtype):
            mismatched_types.append((column_name, expected_polars_dtype, actual_polars_dtype))

    if mismatched_types:
        message = _build_error_message_for_column_types(mismatched_types)
        raise SchemaError(message)


def _get_polars_dtype(schema: Schema, column_name: str) -> pl.DataType | None:
    return schema.get_column_type(column_name)._polars_data_type


def _build_error_message_for_missing_columns(missing_column: list[str]) -> str:
    return f"The columns {missing_column} are missing."


def _build_error_message_for_additional_columns(additional_columns: list[str]) -> str:
    return f"The columns {additional_columns} are not expected."


def _build_error_message_for_columns_in_wrong_order(expected: list[str], actual: list[str]) -> str:
    result = "The columns are in the wrong order:\n"
    result += f"    Expected: {expected}\n"
    result += f"    Actual:   {actual}"
    return result


def _build_error_message_for_column_types(mismatched_types: list[tuple[str, pl.DataType, pl.DataType]]) -> str:
    result = "The following columns have the wrong type:"
    for column_name, expected_type, actual_type in mismatched_types:
        result += f"\n    - '{column_name}': Expected '{expected_type}', but got '{actual_type}'."

    return result
