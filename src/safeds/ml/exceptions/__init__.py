"""Custom exceptions that can be raised when working with ML models."""

from ._exceptions import (
    DatasetContainsTargetError,
    DatasetMissesFeaturesError,
    LearningError,
    ModelNotFittedError,
    PredictionError,
)

__all__ = [
    "DatasetContainsTargetError",
    "DatasetMissesFeaturesError",
    "LearningError",
    "ModelNotFittedError",
    "PredictionError",
]
