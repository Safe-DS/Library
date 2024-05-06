"""Classes for classification tasks."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._convolutional2d_layer import Convolutional2DLayer, ConvolutionalTranspose2DLayer
    from ._flatten_layer import FlattenLayer
    from ._forward_layer import ForwardLayer
    from ._input_conversion import InputConversion
    from ._input_conversion_image import InputConversionImage
    from ._input_conversion_table import InputConversionTable
    from ._input_conversion_time_series import InputConversionTimeSeries
    from ._layer import Layer
    from ._lstm_layer import LSTMLayer
    from ._model import NeuralNetworkClassifier, NeuralNetworkRegressor
    from ._output_conversion import OutputConversion
    from ._output_conversion_image import (
        OutputConversionImageToColumn,
        OutputConversionImageToImage,
        OutputConversionImageToTable,
    )
    from ._output_conversion_table import OutputConversionTable
    from ._output_conversion_time_series import OutputConversionTimeSeries
    from ._pooling2d_layer import AvgPooling2DLayer, MaxPooling2DLayer

apipkg.initpkg(
    __name__,
    {
        "AvgPooling2DLayer": "._pooling2d_layer:AvgPooling2DLayer",
        "Convolutional2DLayer": "._convolutional2d_layer:Convolutional2DLayer",
        "ConvolutionalTranspose2DLayer": "._convolutional2d_layer:ConvolutionalTranspose2DLayer",
        "FlattenLayer": "._flatten_layer:FlattenLayer",
        "ForwardLayer": "._forward_layer:ForwardLayer",
        "InputConversion": "._input_conversion:InputConversion",
        "InputConversionImage": "._input_conversion_image:InputConversionImage",
        "InputConversionTable": "._input_conversion_table:InputConversionTable",
        "Layer": "._layer:Layer",
        "OutputConversion": "._output_conversion:OutputConversion",
        "InputConversionTimeSeries": "._input_conversion_time_series:InputConversionTimeSeries",
        "LSTMLayer": "._lstm_layer:LSTMLayer",
        "OutputConversionTable": "._output_conversion_table:OutputConversionTable",
        "OutputConversionTimeSeries": "._output_conversion_time_series:OutputConversionTimeSeries",
        "MaxPooling2DLayer": "._pooling2d_layer:MaxPooling2DLayer",
        "NeuralNetworkClassifier": "._model:NeuralNetworkClassifier",
        "NeuralNetworkRegressor": "._model:NeuralNetworkRegressor",
        "OutputConversionImageToColumn": "._output_conversion_image:OutputConversionImageToColumn",
        "OutputConversionImageToImage": "._output_conversion_image:OutputConversionImageToImage",
        "OutputConversionImageToTable": "._output_conversion_image:OutputConversionImageToTable",
    },
)

__all__ = [
    "AvgPooling2DLayer",
    "Convolutional2DLayer",
    "ConvolutionalTranspose2DLayer",
    "FlattenLayer",
    "ForwardLayer",
    "InputConversion",
    "InputConversionImage",
    "InputConversionTable",
    "Layer",
    "MaxPooling2DLayer",
    "OutputConversion",
    "InputConversionTimeSeries",
    "LSTMLayer",
    "OutputConversionTable",
    "OutputConversionTimeSeries",
    "NeuralNetworkClassifier",
    "NeuralNetworkRegressor",
    "OutputConversionImageToColumn",
    "OutputConversionImageToImage",
    "OutputConversionImageToTable",
]
