"""Custom exceptions that can be raised by Safe-DS."""

from safeds.exceptions._data import (
    ColumnIsTargetError,
    ColumnLengthMismatchError,
    ColumnSizeError,
    DuplicateColumnNameError,
    IllegalSchemaModificationError,
    IndexOutOfBoundsError,
    MissingValuesColumnError,
    NonNumericColumnError,
    TransformerNotFittedError,
    UnknownColumnNameError,
    ValueNotPresentWhenFittedError,
    WrongFileExtensionError,
)
from safeds.exceptions._generic import (
    Bound,
    ClosedBound,
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
    "ColumnIsTargetError",
    "ColumnLengthMismatchError",
    "ColumnSizeError",
    "DuplicateColumnNameError",
    "IllegalSchemaModificationError",
    "IndexOutOfBoundsError",
    "MissingValuesColumnError",
    "NonNumericColumnError",
    "TransformerNotFittedError",
    "UnknownColumnNameError",
    "ValueNotPresentWhenFittedError",
    "WrongFileExtensionError",
    # ML exceptions
    "DatasetContainsTargetError",
    "DatasetMissesDataError",
    "DatasetMissesFeaturesError",
    "LearningError",
    "ModelNotFittedError",
    "PredictionError",
    "UntaggedTableError",
    # Other
    "Bound",
    "ClosedBound",
    "OpenBound",
]
