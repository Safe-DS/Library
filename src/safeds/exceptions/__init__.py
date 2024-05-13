"""Custom exceptions that can be raised by Safe-DS."""

from ._data import (
    ColumnLengthMismatchError,
    ColumnSizeError,
    DuplicateColumnNameError,
    DuplicateIndexError,
    FileExtensionError,
    IllegalFormatError,
    IndexOutOfBoundsError,
    MissingValuesColumnError,
    NonNumericColumnError,
    OutputLengthMismatchError,
    TransformerNotFittedError,
    ValueNotPresentWhenFittedError,
)
from ._ml import (
    DatasetMissesDataError,
    DatasetMissesFeaturesError,
    FeatureDataMismatchError,
    InputSizeError,
    InvalidModelStructureError,
    LearningError,
    ModelNotFittedError,
    PlainTableError,
    PredictionError,
)


class SafeDsError(Exception):
    """Base class for all exceptions raised by Safe-DS."""


class ColumnNotFoundError(SafeDsError):
    """Exception raised when trying to access an invalid column name."""


class OutOfBoundsError(SafeDsError):
    """Exception raised when a value is outside its expected range."""


__all__ = [
    "SafeDsError",
    "ColumnNotFoundError",
    "OutOfBoundsError",
    # TODO
    # Data exceptions
    "ColumnLengthMismatchError",
    "ColumnSizeError",
    "DuplicateColumnNameError",
    "DuplicateIndexError",
    "IllegalFormatError",
    "IndexOutOfBoundsError",
    "MissingValuesColumnError",
    "NonNumericColumnError",
    "OutputLengthMismatchError",
    "TransformerNotFittedError",
    "ValueNotPresentWhenFittedError",
    "FileExtensionError",
    # ML exceptions
    "DatasetMissesDataError",
    "DatasetMissesFeaturesError",
    "FeatureDataMismatchError",
    "InputSizeError",
    "InvalidModelStructureError",
    "LearningError",
    "ModelNotFittedError",
    "PlainTableError",
    "PredictionError",
]
