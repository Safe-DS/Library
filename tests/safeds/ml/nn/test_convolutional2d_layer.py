from typing import Literal, Type

import pytest
from torch import nn

from safeds.data.image.typing import ImageSize
from safeds.ml.nn import Convolutional2DLayer, ConvolutionalTranspose2DLayer


class TestConvolutional2DLayer:

    @pytest.mark.parametrize(
        ("activation_function", "activation_layer"),
        [
            ("sigmoid", nn.Sigmoid),
            ("relu", nn.ReLU),
            ("softmax", nn.Softmax)
        ],
    )
    @pytest.mark.parametrize(
        ("conv_type", "torch_layer", "output_channel", "kernel_size", "stride", "padding", "out_channel", "out_width", "out_height"),
        [
            (Convolutional2DLayer, nn.Conv2d, 30, 2, 2, 2, 30, 7, 12),
            (ConvolutionalTranspose2DLayer, nn.ConvTranspose2d, 30, 2, 2, 2, 30, 16, 36),
        ],
    )
    def test_should_create_pooling_layer(self, activation_function: Literal["sigmoid", "relu", "softmax"], activation_layer: Type[nn.Module], conv_type: Type[Convolutional2DLayer], torch_layer: Type[nn.Module], output_channel: int, kernel_size: int, stride: int, padding: int, out_channel: int, out_width: int, out_height: int) -> None:
        layer = conv_type(output_channel, kernel_size, stride=stride, padding=padding)
        input_size = ImageSize(10, 20, 30, _ignore_invalid_channel=True)
        layer._set_input_size(input_size)
        assert layer.input_size == input_size
        assert layer.output_size == ImageSize(out_width, out_height, out_channel, _ignore_invalid_channel=True)
        modules = list(next(layer._get_internal_layer(activation_function=activation_function).modules()).children())
        assert isinstance(modules[0], torch_layer)
        assert isinstance(modules[1], activation_layer)

    @pytest.mark.parametrize(
        "activation_function",
        [
            "sigmoid",
            "relu",
            "softmax",
        ],
    )
    @pytest.mark.parametrize(
        ("conv_type", "torch_layer", "output_channel", "kernel_size", "stride", "padding", "out_channel", "out_width", "out_height"),
        [
            (Convolutional2DLayer, nn.Conv2d, 30, 2, 2, 2, 30, 7, 12),
            (ConvolutionalTranspose2DLayer, nn.ConvTranspose2d, 30, 2, 2, 2, 30, 16, 36),
        ],
    )
    def test_should_raise_if_input_size_not_set(self, activation_function: Literal["sigmoid", "relu", "softmax"], conv_type: Type[Convolutional2DLayer], torch_layer: Type[nn.Module], output_channel: int, kernel_size: int, stride: int, padding: int, out_channel: int, out_width: int, out_height: int) -> None:
        layer = conv_type(output_channel, kernel_size, stride=stride, padding=padding)
        with pytest.raises(ValueError, match=r"The input_size is not yet set."):
            layer.input_size
        with pytest.raises(ValueError, match=r"The input_size is not yet set. The layer cannot compute the output_size if the input_size is not set."):
            layer.output_size
        with pytest.raises(ValueError, match=r"The input_size is not yet set. The internal layer can only be created when the input_size is set."):
            layer._get_internal_layer(activation_function)

    @pytest.mark.parametrize(
        ("conv_type", "torch_layer", "output_channel", "kernel_size", "stride", "padding", "out_channel", "out_width", "out_height"),
        [
            (Convolutional2DLayer, nn.Conv2d, 30, 2, 2, 2, 30, 7, 12),
            (ConvolutionalTranspose2DLayer, nn.ConvTranspose2d, 30, 2, 2, 2, 30, 16, 36),
        ],
    )
    def test_should_raise_if_input_size_is_set_with_int(self, conv_type: Type[Convolutional2DLayer], torch_layer: Type[nn.Module], output_channel: int, kernel_size: int, stride: int, padding: int, out_channel: int, out_width: int, out_height: int) -> None:
        layer = conv_type(output_channel, kernel_size, stride=stride, padding=padding)
        with pytest.raises(ValueError, match=r"The input_size of a convolution layer has to be of type ImageSize."):
            layer._set_input_size(1)
