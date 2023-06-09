"""Custom exceptions that can be raised by Safe-DS."""

from safeds.exceptions._data import (
    ColumnLengthMismatchError,
    ColumnSizeError,
    DuplicateColumnNameError,
    IndexOutOfBoundsError,
    NonNumericColumnError,
    SchemaMismatchError,
    TransformerNotFittedError,
    UnknownColumnNameError,
    ValueNotPresentWhenFittedError,
    WrongFileExtensionError,
    MissingValuesColumnError,
)
from safeds.exceptions._ml import (
    DatasetContainsTargetError,
    DatasetMissesFeaturesError,
    LearningError,
    ModelNotFittedError,
    PredictionError,
    UntaggedTableError,
    DatasetMissesDataError,
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
    "MissingValuesColumnError",
    # ML exceptions
    "DatasetContainsTargetError",
    "DatasetMissesFeaturesError",
    "LearningError",
    "ModelNotFittedError",
    "PredictionError",
    "UntaggedTableError",
    "DatasetMissesDataError",
]
