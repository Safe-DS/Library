"""Classes for classification tasks."""

from ._fnn_layer import FNNLayer
from ._model import ClassificationNeuralNetwork, RegressionNeuralNetwork

__all__ = [
    "FNNLayer",
    "ClassificationNeuralNetwork",
    "RegressionNeuralNetwork",
]
