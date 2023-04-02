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
    TransformerNotFittedError,
    UnknownColumnNameError,
)
from ._ml_exceptions import (
    DatasetContainsTargetError,
    DatasetMissesFeaturesError,
    LearningError,
    ModelNotFittedError,
    PredictionError,
)

__all__ = [
    "ColumnLengthMismatchError",
    "ColumnSizeError",
    "DatasetContainsTargetError",
    "DatasetMissesFeaturesError",
    "DuplicateColumnNameError",
    "IndexOutOfBoundsError",
    "LearningError",
    "MissingDataError",
    "MissingSchemaError",
    "ModelNotFittedError",
    "NonNumericColumnError",
    "PredictionError",
    "SchemaMismatchError",
    "TransformerNotFittedError",
    "UnknownColumnNameError",
]
