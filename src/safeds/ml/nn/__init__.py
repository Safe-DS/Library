"""Classes for classification tasks."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._forward_layer import ForwardLayer
    from ._model import NeuralNetworkClassifier, NeuralNetworkRegressor

apipkg.initpkg(
    __name__,
    {
        "ForwardLayer": "._forward_layer:ForwardLayer",
        "NeuralNetworkClassifier": "._model:NeuralNetworkClassifier",
        "NeuralNetworkRegressor": "._model:NeuralNetworkRegressor",
    },
)

__all__ = [
    "ForwardLayer",
    "NeuralNetworkClassifier",
    "NeuralNetworkRegressor",
]
