"""Custom exceptions that can be raised by Safe-DS."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._data import (
    DuplicateIndexError,
    IllegalFormatError,
    MissingValuesColumnError,
    NonNumericColumnError,
    OutputLengthMismatchError,
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

if TYPE_CHECKING:
    from safeds.data.tabular.transformation import TableTransformer


class SafeDsError(Exception):
    """Base class for all exceptions defined by Safe-DS."""


class ColumnNotFoundError(SafeDsError, IndexError):
    """Raised when trying to access an invalid column name."""


class ColumnTypeError(SafeDsError, TypeError):
    """Raised when a column has the wrong type."""


class DuplicateColumnError(SafeDsError, ValueError):
    """Raised when a table has duplicate column names."""


class FileExtensionError(SafeDsError, ValueError):
    """Raised when a path has the wrong file extension."""


class IndexOutOfBoundsError(IndexError):
    """Raised when trying to access an invalid index."""


class LazyComputationError(SafeDsError, RuntimeError):
    """Raised when a lazy computation fails."""


class LengthMismatchError(SafeDsError, ValueError):
    """Raised when objects have different lengths."""


class MissingValuesError(Exception):
    """Raised when an operation cannot be performed on missing values."""


class NotFittedError(SafeDsError, RuntimeError):
    """Raised when an object (e.g. a transformer or model) is not fitted."""

    def __init__(self, *, kind: str = "object") -> None:
        super().__init__(f"This {kind} has not been fitted yet.")


class NotInvertibleError(SafeDsError, TypeError):
    """Raised when inverting a non-invertible transformation."""

    def __init__(self, transformer: TableTransformer) -> None:
        super().__init__(f"A {transformer.__class__.__name__} is not invertible.")


class OutOfBoundsError(SafeDsError, ValueError):
    """Raised when a value is outside its expected range."""


class SchemaError(SafeDsError, TypeError):
    """Raised when tables have incompatible schemas."""


__all__ = [  # noqa: RUF022
    "SafeDsError",
    "ColumnNotFoundError",
    "ColumnTypeError",
    "DuplicateColumnError",
    "FileExtensionError",
    "IndexOutOfBoundsError",
    "LazyComputationError",
    "LengthMismatchError",
    "MissingValuesError",
    "NotFittedError",
    "NotInvertibleError",
    "OutOfBoundsError",
    "SchemaError",
    # TODO
    # Data exceptions
    "DuplicateIndexError",
    "IllegalFormatError",
    "MissingValuesColumnError",
    "NonNumericColumnError",
    "OutputLengthMismatchError",
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
    "PlainTableError",
    "PredictionError",
    "TargetDataMismatchError",
]
