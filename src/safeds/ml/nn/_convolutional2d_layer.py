from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

from safeds.data.image.typing import ImageSize

if TYPE_CHECKING:
    from torch import Tensor, nn

from safeds.ml.nn._layer import _Layer


def _create_internal_model(input_size: int, output_size: int, kernel_size: int, activation_function: Literal["sigmoid", "relu", "softmax"], padding: int, stride: int, transpose: bool, output_padding: int = 0) -> nn.Module:
    from torch import nn

    class _InternalLayer(nn.Module):
        def __init__(self, input_size: int, output_size: int, kernel_size: int, activation_function: Literal["sigmoid", "relu", "softmax"], padding: int, stride: int, transpose: bool, output_padding: int):
            super().__init__()
            if transpose:
                self._layer = nn.ConvTranspose2d(in_channels=input_size, out_channels=output_size, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=output_padding)
            else:
                self._layer = nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=kernel_size, padding=padding, stride=stride)
            match activation_function:
                case "sigmoid":
                    self._fn = nn.Sigmoid()
                case "relu":
                    self._fn = nn.ReLU()
                case "softmax":
                    self._fn = nn.Softmax()

        def forward(self, x: Tensor) -> Tensor:
            return self._fn(self._layer(x))

    return _InternalLayer(input_size, output_size, kernel_size, activation_function, padding, stride, transpose, output_padding)


class Convolutional2DLayer(_Layer):
    def __init__(self, output_channel: int, kernel_size: int, *, stride: int = 1, padding: int = 0):
        """
        Create a Convolutional 2D Layer.

        Parameters
        ----------
        output_channel:
            the amount of output channels
        kernel_size:
            the size of the kernel
        stride:
            the stride of the convolution
        padding:
            the padding of the convolution
        """
        self._output_channel = output_channel
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._output_size = None
        self._input_size = None

    def _get_internal_layer(self, *, activation_function: Literal["sigmoid", "relu", "softmax"]) -> nn.Module:
        return _create_internal_model(self._input_size.channel, self._output_channel, self._kernel_size, activation_function, self._padding, self._stride, transpose=False)

    @property
    def input_size(self) -> ImageSize:
        """
        Get the input_size of this layer.

        Returns
        -------
        result:
            The amount of values being passed into this layer.
        """
        return self._input_size

    @property
    def output_size(self) -> ImageSize:
        """
        Get the output_size of this layer.

        Returns
        -------
        result:
            The Number of Neurons in this layer.
        """
        if self._output_size is None and self._output_size is not None:
            new_width = math.ceil((self._input_size.width + self._padding * 2 - self._kernel_size + 1) / (1.0 * self._stride))
            new_height = math.ceil((self._input_size.height + self._padding * 2 - self._kernel_size + 1) / (1.0 * self._stride))
            self._output_size = ImageSize(new_width, new_height, self._output_channel, _ignore_invalid_channel=True)
        return self._output_size

    def _set_input_size(self, input_size: ImageSize) -> None:
        self._input_size = input_size
        self._output_size = None


class ConvolutionalTranspose2DLayer(Convolutional2DLayer):

    def __init__(self, output_channel: int, kernel_size: int, *, stride: int = 1, padding: int = 0, output_padding: int = 0):
        """
        Create a Convolutional Transpose 2D Layer.

        Parameters
        ----------
        output_channel:
            the amount of output channels
        kernel_size:
            the size of the kernel
        stride:
            the stride of the transposed convolution
        padding:
            the padding of the transposed convolution
        output_padding:
            the output padding of the transposed convolution
        """
        super().__init__(output_channel, kernel_size, stride=stride, padding=padding)
        self._output_padding = output_padding

    def _get_internal_layer(self, *, activation_function: Literal["sigmoid", "relu", "softmax"]) -> nn.Module:
        return _create_internal_model(self._input_size.channel, self._output_channel, self._kernel_size, activation_function, self._padding, self._stride, transpose=True, output_padding=self._output_padding)

    @property
    def output_size(self) -> ImageSize:
        if self._output_size is None and self._output_size is not None:
            new_width = (self.input_size.width - 1) * self._stride - 2 * self._padding + self._kernel_size + self._output_padding
            new_height = (self.input_size.height - 1) * self._stride - 2 * self._padding + self._kernel_size + self._output_padding
            self._output_size = ImageSize(new_width, new_height, self._output_channel, _ignore_invalid_channel=True)
        return self._output_size
