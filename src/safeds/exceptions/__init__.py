"""Custom exceptions that can be raised by Safe-DS."""

from safeds.exceptions._data import (
    ColumnIsTaggedError,
    ColumnLengthMismatchError,
    ColumnSizeError,
    DuplicateColumnNameError,
    IllegalSchemaModificationError,
    IndexOutOfBoundsError,
    NonNumericColumnError,
    SchemaMismatchError,
    TransformerNotFittedError,
    UnknownColumnNameError,
    ValueNotPresentWhenFittedError,
    WrongFileExtensionError,
)
from safeds.exceptions._ml import (
    DatasetContainsTargetError,
    DatasetMissesFeaturesError,
    LearningError,
    ModelNotFittedError,
    PredictionError,
    UntaggedTableError,
)

__all__ = [
    # Data exceptions
    "ColumnLengthMismatchError",
    "ColumnSizeError",
    "DuplicateColumnNameError",
    "IndexOutOfBoundsError",
    "NonNumericColumnError",
    "SchemaMismatchError",
    "TransformerNotFittedError",
    "UnknownColumnNameError",
    "ValueNotPresentWhenFittedError",
    "WrongFileExtensionError",
    "IllegalSchemaModificationError",
    "ColumnIsTaggedError",
    # ML exceptions
    "DatasetContainsTargetError",
    "DatasetMissesFeaturesError",
    "LearningError",
    "ModelNotFittedError",
    "PredictionError",
    "UntaggedTableError",
]
