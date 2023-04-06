"""Custom exceptions that can be raised when working with tabular data."""

from ._exceptions import (
    ColumnLengthMismatchError,
    ColumnSizeError,
    DuplicateColumnNameError,
    IndexOutOfBoundsError,
    MissingDataError,
    MissingSchemaError,
    NonNumericColumnError,
    SchemaMismatchError,
    TransformerNotFittedError,
    UnknownColumnNameError,
)

__all__ = [
    "ColumnLengthMismatchError",
    "ColumnSizeError",
    "DuplicateColumnNameError",
    "IndexOutOfBoundsError",
    "MissingDataError",
    "MissingSchemaError",
    "NonNumericColumnError",
    "SchemaMismatchError",
    "TransformerNotFittedError",
    "UnknownColumnNameError",
]
