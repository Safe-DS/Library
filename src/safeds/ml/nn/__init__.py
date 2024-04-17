"""Classes for classification tasks."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._forward_layer import ForwardLayer
    from ._input_conversion_table import InputConversionTable
    from ._model import NeuralNetworkClassifier, NeuralNetworkRegressor
    from ._output_conversion_table import OutputConversionTable

apipkg.initpkg(
    __name__,
    {
        "ForwardLayer": "._forward_layer:ForwardLayer",
        "InputConversionTable": "._input_conversion_table:InputConversionTable",
        "OutputConversionTable": "._output_conversion_table:OutputConversionTable",
        "NeuralNetworkClassifier": "._model:NeuralNetworkClassifier",
        "NeuralNetworkRegressor": "._model:NeuralNetworkRegressor",
    },
)

__all__ = [
    "ForwardLayer",
    "InputConversionTable",
    "OutputConversionTable",
    "NeuralNetworkClassifier",
    "NeuralNetworkRegressor",
]
