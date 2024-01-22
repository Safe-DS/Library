"""Classes for classification tasks."""

from ._fnn_layer import FNNLayer
from ._model import ClassificationModel, RegressionModel

__all__ = [
    "FNNLayer",
    "ClassificationModel",
    "RegressionModel",
]
