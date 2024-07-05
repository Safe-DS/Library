"""Layers of neural networks."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._convolutional2d_layer import Convolutional2DLayer, ConvolutionalTranspose2DLayer
    from ._dropout_layer import DropoutLayer
    from ._flatten_layer import FlattenLayer
    from ._forward_layer import ForwardLayer
    from ._gru_layer import GRULayer
    from ._layer import Layer
    from ._lstm_layer import LSTMLayer
    from ._pooling2d_layer import AveragePooling2DLayer, MaxPooling2DLayer

apipkg.initpkg(
    __name__,
    {
        "Convolutional2DLayer": "._convolutional2d_layer:Convolutional2DLayer",
        "ConvolutionalTranspose2DLayer": "._convolutional2d_layer:ConvolutionalTranspose2DLayer",
        "FlattenLayer": "._flatten_layer:FlattenLayer",
        "ForwardLayer": "._forward_layer:ForwardLayer",
        "Layer": "._layer:Layer",
        "LSTMLayer": "._lstm_layer:LSTMLayer",
        "GRULayer": "._gru_layer:GRULayer",
        "AveragePooling2DLayer": "._pooling2d_layer:AveragePooling2DLayer",
        "MaxPooling2DLayer": "._pooling2d_layer:MaxPooling2DLayer",
        "DropoutLayer": "._dropout_layer:DropoutLayer",
    },
)

__all__ = [
    "Convolutional2DLayer",
    "ConvolutionalTranspose2DLayer",
    "FlattenLayer",
    "ForwardLayer",
    "Layer",
    "LSTMLayer",
    "GRULayer",
    "AveragePooling2DLayer",
    "MaxPooling2DLayer",
    "DropoutLayer",
]
