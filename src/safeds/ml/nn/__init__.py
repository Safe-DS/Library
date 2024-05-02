"""Classes for classification tasks."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._forward_layer import ForwardLayer
    from ._input_conversion import InputConversion
    from ._input_conversion_table import InputConversionTable
    from ._layer import Layer
    from ._model import NeuralNetworkClassifier, NeuralNetworkRegressor
    from ._output_conversion import OutputConversion
    from ._output_conversion_table import OutputConversionTable

apipkg.initpkg(
    __name__,
    {
        "ForwardLayer": "._forward_layer:ForwardLayer",
        "InputConversion": "._input_conversion:InputConversion",
        "InputConversionTable": "._input_conversion_table:InputConversionTable",
        "Layer": "._layer:Layer",
        "OutputConversion": "._output_conversion:OutputConversion",
        "OutputConversionTable": "._output_conversion_table:OutputConversionTable",
        "NeuralNetworkClassifier": "._model:NeuralNetworkClassifier",
        "NeuralNetworkRegressor": "._model:NeuralNetworkRegressor",
    },
)

__all__ = [
    "ForwardLayer",
    "InputConversion",
    "InputConversionTable",
    "Layer",
    "OutputConversion",
    "OutputConversionTable",
    "NeuralNetworkClassifier",
    "NeuralNetworkRegressor",
]
