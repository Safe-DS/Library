"""Custom exceptions that can be raised by Safe-DS."""

from ._data import (
    DuplicateIndexError,
    IllegalFormatError,
    IndexOutOfBoundsError,
    MissingValuesColumnError,
    NonNumericColumnError,
    OutputLengthMismatchError,
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


class LengthMismatchError(SafeDsError, ValueError):
    """Exception raised when objects have different lengths."""


class NotFittedError(SafeDsError, ValueError):
    """Exception raised when an object (e.g. a transformer or model) is not fitted."""

    def __init__(self, *, kind: str = "object") -> None:
        super().__init__(f"This {kind} has not been fitted yet.")


class OutOfBoundsError(SafeDsError, ValueError):
    """Exception raised when a value is outside its expected range."""


class SchemaError(SafeDsError, TypeError):
    """Exception raised when tables have incompatible schemas."""


# class NotFittedError(Exception):
#     """Raised when a transformer is used before fitting it."""
#
#     def __init__(self) -> None:
#         super().__init__("The transformer has not been fitted yet.")


class ModelNotFittedError(RuntimeError):
    """Raised when a model is used before fitting it."""

    def __init__(self) -> None:
        super().__init__("The model has not been fitted yet.")


__all__ = [
    "SafeDsError",
    "ColumnNotFoundError",
    "ColumnTypeError",
    "DuplicateColumnError",
    "FileExtensionError",
    "LengthMismatchError",
    "NotFittedError",
    "OutOfBoundsError",
    "SchemaError",
    # TODO
    # Data exceptions
    "DuplicateIndexError",
    "IllegalFormatError",
    "IndexOutOfBoundsError",
    "MissingValuesColumnError",
    "NonNumericColumnError",
    "OutputLengthMismatchError",
    "NotFittedError",
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
