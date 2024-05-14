"""Custom exceptions that can be raised by Safe-DS."""

from ._data import (
    ColumnLengthMismatchError,
    ColumnSizeError,
    DuplicateColumnError,
    DuplicateIndexError,
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
    """Base class for all exceptions defined by Safe-DS."""


class ColumnNotFoundError(SafeDsError):
    """Exception raised when trying to access an invalid column name."""


class ColumnTypeError(SafeDsError):
    """Exception raised when a column has the wrong type."""


class FileExtensionError(SafeDsError):
    """Exception raised when a path has the wrong file extension."""


class OutOfBoundsError(SafeDsError):
    """Exception raised when a value is outside its expected range."""


__all__ = [
    "SafeDsError",
    "ColumnNotFoundError",
    "ColumnTypeError",
    "FileExtensionError",
    "OutOfBoundsError",
    # TODO
    # Data exceptions
    "ColumnLengthMismatchError",
    "ColumnSizeError",
    "DuplicateColumnError",
    "DuplicateIndexError",
    "IllegalFormatError",
    "IndexOutOfBoundsError",
    "MissingValuesColumnError",
    "NonNumericColumnError",
    "OutputLengthMismatchError",
    "TransformerNotFittedError",
    "ValueNotPresentWhenFittedError",
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
