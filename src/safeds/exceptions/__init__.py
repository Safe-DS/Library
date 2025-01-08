"""Custom exceptions that can be raised by Safe-DS."""

from ._data import (
    DuplicateIndexError,
    IllegalFormatError,
    IndexOutOfBoundsError,
    MissingValuesColumnError,
    NonNumericColumnError,
    OutputLengthMismatchError,
    TransformerNotFittedError,
    TransformerNotInvertibleError,
    ValueNotPresentWhenFittedError,
)
from ._ml import (
    DatasetMissesDataError,
    DatasetMissesFeaturesError,
    EmptyChoiceError,
    FeatureDataMismatchError,
    FittingWithChoiceError,
    FittingWithoutChoiceError,
    InputSizeError,
    InvalidFitDataError,
    InvalidModelStructureError,
    LearningError,
    ModelNotFittedError,
    PlainTableError,
    PredictionError,
    TargetDataMismatchError,
)


class SafeDsError(Exception):
    """Base class for all exceptions defined by Safe-DS."""


class ColumnNotFoundError(SafeDsError, IndexError):
    """Exception raised when trying to access an invalid column name."""


class ColumnTypeError(SafeDsError, TypeError):
    """Exception raised when a column has the wrong type."""


class DuplicateColumnError(SafeDsError, ValueError):
    """Exception raised when a table has duplicate column names."""


class FileExtensionError(SafeDsError, ValueError):
    """Exception raised when a path has the wrong file extension."""


class OutOfBoundsError(SafeDsError, ValueError):
    """Exception raised when a value is outside its expected range."""


class RowCountMismatchError(SafeDsError, ValueError):
    """Exception raised when columns or tables have different row counts."""


__all__ = [
    "SafeDsError",
    "ColumnNotFoundError",
    "ColumnTypeError",
    "DuplicateColumnError",
    "FileExtensionError",
    "OutOfBoundsError",
    "RowCountMismatchError",
    # TODO
    # Data exceptions
    "DuplicateIndexError",
    "IllegalFormatError",
    "IndexOutOfBoundsError",
    "MissingValuesColumnError",
    "NonNumericColumnError",
    "OutputLengthMismatchError",
    "TransformerNotFittedError",
    "TransformerNotInvertibleError",
    "ValueNotPresentWhenFittedError",
    # ML exceptions
    "DatasetMissesDataError",
    "DatasetMissesFeaturesError",
    "EmptyChoiceError",
    "FeatureDataMismatchError",
    "FittingWithChoiceError",
    "FittingWithoutChoiceError",
    "InvalidFitDataError",
    "InputSizeError",
    "InvalidModelStructureError",
    "LearningError",
    "ModelNotFittedError",
    "PlainTableError",
    "PredictionError",
    "TargetDataMismatchError",
]
