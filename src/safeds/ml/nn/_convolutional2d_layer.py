from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING, Any, Literal

from safeds._config import _init_default_device
from safeds._utils import _structural_hash
from safeds.data.image.typing import ImageSize

if TYPE_CHECKING:
    from torch import Tensor, nn

from safeds.ml.nn import Layer


def _create_internal_model(
    input_size: int,
    output_size: int,
    kernel_size: int,
    activation_function: Literal["sigmoid", "relu", "softmax"],
    padding: int,
    stride: int,
    transpose: bool,
    output_padding: int = 0,
) -> nn.Module:
    from torch import nn

    _init_default_device()

    class _InternalLayer(nn.Module):
        def __init__(
            self,
            input_size: int,
            output_size: int,
            kernel_size: int,
            activation_function: Literal["sigmoid", "relu", "softmax"],
            padding: int,
            stride: int,
            transpose: bool,
            output_padding: int,
        ):
            super().__init__()
            if transpose:
                self._layer = nn.ConvTranspose2d(
                    in_channels=input_size,
                    out_channels=output_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    output_padding=output_padding,
                )
            else:
                self._layer = nn.Conv2d(
                    in_channels=input_size,
                    out_channels=output_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                )
            match activation_function:
                case "sigmoid":
                    self._fn = nn.Sigmoid()
                case "relu":
                    self._fn = nn.ReLU()
                case "softmax":
                    self._fn = nn.Softmax()

        def forward(self, x: Tensor) -> Tensor:
            return self._fn(self._layer(x))

    return _InternalLayer(
        input_size,
        output_size,
        kernel_size,
        activation_function,
        padding,
        stride,
        transpose,
        output_padding,
    )


class Convolutional2DLayer(Layer):
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
        self._input_size: ImageSize | None = None
        self._output_size: ImageSize | None = None

    def _get_internal_layer(self, **kwargs: Any) -> nn.Module:
        if self._input_size is None:
            raise ValueError(
                "The input_size is not yet set. The internal layer can only be created when the input_size is set.",
            )
        if "activation_function" not in kwargs:
            raise ValueError(
                "The activation_function is not set. The internal layer can only be created when the activation_function is provided in the kwargs.",
            )
        if kwargs.get("activation_function") not in ["sigmoid", "relu", "softmax"]:
            raise ValueError(
                f"The activation_function '{kwargs.get('activation_function')}' is not supported. Please choose one of the following: ['sigmoid', 'relu', 'softmax'].",
            )
        else:
            activation_function: Literal["sigmoid", "relu", "softmax"] = kwargs["activation_function"]
        return _create_internal_model(
            self._input_size.channel,
            self._output_channel,
            self._kernel_size,
            activation_function,
            self._padding,
            self._stride,
            transpose=False,
        )

    @property
    def input_size(self) -> ImageSize:
        """
        Get the input_size of this layer.

        Returns
        -------
        result:
            The amount of values being passed into this layer.

        Raises
        ------
        ValueError
            If the input_size is not yet set
        """
        if self._input_size is None:
            raise ValueError("The input_size is not yet set.")
        return self._input_size

    @property
    def output_size(self) -> ImageSize:
        """
        Get the output_size of this layer.

        Returns
        -------
        result:
            The Number of Neurons in this layer.

        Raises
        ------
        ValueError
            If the input_size is not yet set
        """
        if self._input_size is None:
            raise ValueError(
                "The input_size is not yet set. The layer cannot compute the output_size if the input_size is not set.",
            )
        if self._output_size is None:
            new_width = math.ceil(
                (self._input_size.width + self._padding * 2 - self._kernel_size + 1) / (1.0 * self._stride),
            )
            new_height = math.ceil(
                (self._input_size.height + self._padding * 2 - self._kernel_size + 1) / (1.0 * self._stride),
            )
            self._output_size = ImageSize(new_width, new_height, self._output_channel, _ignore_invalid_channel=True)
        return self._output_size

    def _set_input_size(self, input_size: int | ImageSize) -> None:
        if isinstance(input_size, int):
            raise TypeError("The input_size of a convolution layer has to be of type ImageSize.")
        self._input_size = input_size
        self._output_size = None

    def __hash__(self) -> int:
        """
        Return a deterministic hash value for this convolutional 2d layer.

        Returns
        -------
        hash:
            the hash value
        """
        return _structural_hash(
            self._output_channel,
            self._kernel_size,
            self._stride,
            self._padding,
            self._input_size,
            self._output_size,
        )

    def __eq__(self, other: object) -> bool:
        """
        Compare two convolutional 2d layer.

        Parameters
        ----------
        other:
            The convolutional 2d layer to compare to.

        Returns
        -------
        equals:
            Whether the two convolutional 2d layer are the same.
        """
        if not isinstance(other, Convolutional2DLayer) or isinstance(other, ConvolutionalTranspose2DLayer):
            return NotImplemented
        return (self is other) or (
            self._output_channel == other._output_channel
            and self._kernel_size == other._kernel_size
            and self._stride == other._stride
            and self._padding == other._padding
            and self._input_size == other._input_size
            and self._output_size == other._output_size
        )

    def __sizeof__(self) -> int:
        """
        Return the complete size of this object.

        Returns
        -------
        size:
            Size of this object in bytes.
        """
        return (
            sys.getsizeof(self._output_channel)
            + sys.getsizeof(self._kernel_size)
            + sys.getsizeof(self._stride)
            + sys.getsizeof(self._padding)
            + sys.getsizeof(self._input_size)
            + sys.getsizeof(self._output_size)
        )


class ConvolutionalTranspose2DLayer(Convolutional2DLayer):

    def __init__(
        self,
        output_channel: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
    ):
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

    def _get_internal_layer(self, **kwargs: Any) -> nn.Module:
        if self._input_size is None:
            raise ValueError(
                "The input_size is not yet set. The internal layer can only be created when the input_size is set.",
            )
        if "activation_function" not in kwargs:
            raise ValueError(
                "The activation_function is not set. The internal layer can only be created when the activation_function is provided in the kwargs.",
            )
        if kwargs.get("activation_function") not in ["sigmoid", "relu", "softmax"]:
            raise ValueError(
                f"The activation_function '{kwargs.get('activation_function')}' is not supported. Please choose one of the following: ['sigmoid', 'relu', 'softmax'].",
            )
        else:
            activation_function: Literal["sigmoid", "relu", "softmax"] = kwargs["activation_function"]
        return _create_internal_model(
            self._input_size.channel,
            self._output_channel,
            self._kernel_size,
            activation_function,
            self._padding,
            self._stride,
            transpose=True,
            output_padding=self._output_padding,
        )

    @property
    def output_size(self) -> ImageSize:
        if self._input_size is None:
            raise ValueError(
                "The input_size is not yet set. The layer cannot compute the output_size if the input_size is not set.",
            )
        if self._output_size is None:
            new_width = (
                (self.input_size.width - 1) * self._stride
                - 2 * self._padding
                + self._kernel_size
                + self._output_padding
            )
            new_height = (
                (self.input_size.height - 1) * self._stride
                - 2 * self._padding
                + self._kernel_size
                + self._output_padding
            )
            self._output_size = ImageSize(new_width, new_height, self._output_channel, _ignore_invalid_channel=True)
        return self._output_size

    def __hash__(self) -> int:
        """
        Return a deterministic hash value for this convolutional transpose 2d layer.

        Returns
        -------
        hash:
            the hash value
        """
        return _structural_hash(super().__hash__(), self._output_padding)

    def __eq__(self, other: object) -> bool:
        """
        Compare two convolutional transpose 2d layer.

        Parameters
        ----------
        other:
            The convolutional transpose 2d layer to compare to.

        Returns
        -------
        equals:
            Whether the two convolutional transpose 2d layer are the same.
        """
        if not isinstance(other, ConvolutionalTranspose2DLayer):
            return NotImplemented
        return (self is other) or (
            self._output_channel == other._output_channel
            and self._kernel_size == other._kernel_size
            and self._stride == other._stride
            and self._padding == other._padding
            and self._input_size == other._input_size
            and self._output_size == other._output_size
            and self._output_padding == other._output_padding
        )

    def __sizeof__(self) -> int:
        """
        Return the complete size of this object.

        Returns
        -------
        size:
            Size of this object in bytes.
        """
        return sys.getsizeof(self._output_padding) + super().__sizeof__()
