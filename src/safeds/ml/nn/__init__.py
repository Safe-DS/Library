"""Classes for classification tasks."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._pooling2d_layer import AvgPooling2DLayer
    from ._convolutional2d_layer import Convolutional2DLayer
    from ._flatten_layer import FlattenLayer
    from ._forward_layer import ForwardLayer
    from ._input_conversion_image import InputConversionImage
    from ._input_conversion_table import InputConversionTable
    from ._pooling2d_layer import MaxPooling2DLayer
    from ._model import NeuralNetworkClassifier, NeuralNetworkRegressor
    from ._output_conversion_image import OutputConversionImageToImage, OutputConversionImageToTable
    from ._output_conversion_table import OutputConversionTable

apipkg.initpkg(
    __name__,
    {
        "AvgPooling2DLayer": "._pooling2d_layer:AvgPooling2DLayer",
        "Convolutional2DLayer": "._convolutional2d_layer:Convolutional2DLayer",
        "FlattenLayer": "._flatten_layer:FlattenLayer",
        "ForwardLayer": "._forward_layer:ForwardLayer",
        "InputConversionImage": "._input_conversion_image:InputConversionImage",
        "InputConversionTable": "._input_conversion_table:InputConversionTable",
        "MaxPooling2DLayer": "._pooling2d_layer:MaxPooling2DLayer",
        "NeuralNetworkClassifier": "._model:NeuralNetworkClassifier",
        "NeuralNetworkRegressor": "._model:NeuralNetworkRegressor",
        "OutputConversionImageToImage": "._output_conversion_image:OutputConversionImageToImage",
        "OutputConversionImageToTable": "._output_conversion_image:OutputConversionImageToTable",
        "OutputConversionTable": "._output_conversion_table:OutputConversionTable",
        "Pooling2DLayer": "._pooling2d_layer:Pooling2DLayer",
    },
)

__all__ = [
    "AvgPooling2DLayer",
    "Convolutional2DLayer",
    "FlattenLayer",
    "ForwardLayer",
    "InputConversionImage",
    "InputConversionTable",
    "MaxPooling2DLayer",
    "NeuralNetworkClassifier",
    "NeuralNetworkRegressor",
    "OutputConversionImageToImage",
    "OutputConversionImageToTable",
    "OutputConversionTable",
]
