"""Custom exceptions that can be raised by the `safe-ds` package."""

from ._data_exceptions import (
    ColumnLengthMismatchError,
    ColumnSizeError,
    DuplicateColumnNameError,
    IndexOutOfBoundsError,
    MissingDataError,
    MissingSchemaError,
    NonNumericColumnError,
    SchemaMismatchError,
    UnknownColumnNameError,
)
from ._learning_exceptions import LearningError, NotFittedError, PredictionError

__all__ = [
    "ColumnLengthMismatchError",
    "ColumnSizeError",
    "DuplicateColumnNameError",
    "IndexOutOfBoundsError",
    "LearningError",
    "MissingDataError",
    "MissingSchemaError",
    "NonNumericColumnError",
    "NotFittedError",
    "PredictionError",
    "SchemaMismatchError",
    "UnknownColumnNameError",
]
