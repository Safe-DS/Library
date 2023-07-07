"""Custom exceptions that can be raised by Safe-DS."""

from safeds.exceptions._data import (
    ColumnLengthMismatchError,
    ColumnSizeError,
    DuplicateColumnNameError,
    IndexOutOfBoundsError,
    MissingValuesColumnError,
    NonNumericColumnError,
    SchemaMismatchError,
    TransformerNotFittedError,
    UnknownColumnNameError,
    ValueNotPresentWhenFittedError,
    WrongFileExtensionError,
)
from safeds.exceptions._generic import (
    Bound,
    ClosedBound,
    Infinity,
    MinInfinity,
    OpenBound,
    OutOfBoundsError,
)
from safeds.exceptions._ml import (
    DatasetContainsTargetError,
    DatasetMissesDataError,
    DatasetMissesFeaturesError,
    LearningError,
    ModelNotFittedError,
    PredictionError,
    UntaggedTableError,
)

__all__ = [
    # Generic exceptions
    "OutOfBoundsError",
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
    # Other
    "Bound",
    "ClosedBound",
    "Infinity",
    "MinInfinity",
    "OpenBound",
]
