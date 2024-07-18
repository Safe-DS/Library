from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING, Any, Literal

from safeds._utils import _structural_hash

from ._layer import Layer

if TYPE_CHECKING:
    from torch import nn

    from safeds.ml.nn.typing import ModelImageSize


class Convolutional2DLayer(Layer):
    """
    A convolutional 2D Layer.

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

    def __init__(self, output_channel: int, kernel_size: int, *, stride: int = 1, padding: int = 0):
        self._output_channel = output_channel
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._input_size: ModelImageSize | None = None
        self._output_size: ModelImageSize | None = None

    def _get_internal_layer(self, **kwargs: Any) -> nn.Module:
        from ._internal_layers import _InternalConvolutional2DLayer  # slow import on global level

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
        return _InternalConvolutional2DLayer(
            self._input_size.channel,
            self._output_channel,
            self._kernel_size,
            activation_function,
            self._padding,
            self._stride,
            transpose=False,
            output_padding=0,
        )

    @property
    def input_size(self) -> ModelImageSize:
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
    def output_size(self) -> ModelImageSize:
        """
        Get the output_size of this layer.

        Returns
        -------
        result:
            The number of neurons in this layer.

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
            self._output_size = self._input_size.__class__(
                new_width,
                new_height,
                self._output_channel,
                _ignore_invalid_channel=True,
            )
        return self._output_size

    def _set_input_size(self, input_size: int | ModelImageSize) -> None:
        if isinstance(input_size, int):
            raise TypeError("The input_size of a convolution layer has to be of type ImageSize.")
        self._input_size = input_size
        self._output_size = None

    def _contains_choices(self) -> bool:
        return False

    def _get_layers_for_all_choices(self) -> list[Convolutional2DLayer]:
        raise NotImplementedError  # pragma: no cover

    def __hash__(self) -> int:
        return _structural_hash(
            self._output_channel,
            self._kernel_size,
            self._stride,
            self._padding,
            self._input_size,
            self._output_size,
        )

    def __eq__(self, other: object) -> bool:
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
        return (
            sys.getsizeof(self._output_channel)
            + sys.getsizeof(self._kernel_size)
            + sys.getsizeof(self._stride)
            + sys.getsizeof(self._padding)
            + sys.getsizeof(self._input_size)
            + sys.getsizeof(self._output_size)
        )


class ConvolutionalTranspose2DLayer(Convolutional2DLayer):
    """
    A convolutional transpose 2D Layer.

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

    def __init__(
        self,
        output_channel: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
    ):
        super().__init__(output_channel, kernel_size, stride=stride, padding=padding)
        self._output_padding = output_padding

    def _get_internal_layer(self, **kwargs: Any) -> nn.Module:
        from ._internal_layers import _InternalConvolutional2DLayer  # slow import on global level

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
        return _InternalConvolutional2DLayer(
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
    def output_size(self) -> ModelImageSize:
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
            self._output_size = self._input_size.__class__(
                new_width,
                new_height,
                self._output_channel,
                _ignore_invalid_channel=True,
            )
        return self._output_size

    def __hash__(self) -> int:
        return _structural_hash(super().__hash__(), self._output_padding)

    def __eq__(self, other: object) -> bool:
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
        return sys.getsizeof(self._output_padding) + super().__sizeof__()
