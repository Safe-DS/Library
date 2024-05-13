"""Neural networks for various tasks."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._model import NeuralNetworkClassifier, NeuralNetworkRegressor

apipkg.initpkg(
    __name__,
    {
        "NeuralNetworkClassifier": "._model:NeuralNetworkClassifier",
        "NeuralNetworkRegressor": "._model:NeuralNetworkRegressor",
    },
)

__all__ = [
    "NeuralNetworkClassifier",
    "NeuralNetworkRegressor",
]
