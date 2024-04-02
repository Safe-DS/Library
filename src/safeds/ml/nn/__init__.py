"""Classes for classification tasks."""

from ._fnn_layer import FNNLayer
from ._model import NeuralNetworkClassifier, NeuralNetworkRegressor

__all__ = [
    "FNNLayer",
    "NeuralNetworkClassifier",
    "NeuralNetworkRegressor",
]
