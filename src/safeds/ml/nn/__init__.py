"""Classes for classification tasks."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._forward_layer import ForwardLayer
    from ._input_conversion_table import InputConversionTable
    from ._input_conversion_time_series import InputConversionTimeSeries
    from ._lstm_layer import LSTMLayer
    from ._model import NeuralNetworkClassifier, NeuralNetworkRegressor
    from ._output_conversion_table import OutputConversionTable
    from ._output_conversion_time_series import OutputConversionTimeSeries

apipkg.initpkg(
    __name__,
    {
        "ForwardLayer": "._forward_layer:ForwardLayer",
        "InputConversionTable": "._input_conversion_table:InputConversionTable",
        "InputConversionTimeSeries": "._input_conversion_time_series:InputConversionTimeSeries",
        "LSTMLayer": "._lstm_layer:LSTMLayer",
        "OutputConversionTable": "._output_conversion_table:OutputConversionTable",
        "OutputConversionTimeSeries": "._output_conversion_time_series:OutputConversionTimeSeries",
        "NeuralNetworkClassifier": "._model:NeuralNetworkClassifier",
        "NeuralNetworkRegressor": "._model:NeuralNetworkRegressor",
    },
)

__all__ = [
    "ForwardLayer",
    "InputConversionTable",
    "InputConversionTimeSeries",
    "LSTMLayer",
    "OutputConversionTable",
    "OutputConversionTimeSeries",
    "NeuralNetworkClassifier",
    "NeuralNetworkRegressor",
]
