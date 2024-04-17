"""Classes for classification tasks."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._fnn_layer import FNNLayer
    from ._model import NeuralNetworkClassifier, NeuralNetworkRegressor

apipkg.initpkg(__name__, {
    "FNNLayer": "._fnn_layer:FNNLayer",
    "NeuralNetworkClassifier": "._model:NeuralNetworkClassifier",
    "NeuralNetworkRegressor": "._model:NeuralNetworkRegressor",
})

__all__ = [
    "FNNLayer",
    "NeuralNetworkClassifier",
    "NeuralNetworkRegressor",
]
