from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING, Any, Literal

from safeds._utils import _structural_hash
from safeds.ml.nn.typing import ModelImageSize

from ._layer import Layer

if TYPE_CHECKING:
    from torch import nn


class Convolutional1DLayer(Layer):
    """
    A convolutional 1D Layer.

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
        self._input_size: int | None = None
        self._output_size: int | None = None

    def _get_internal_layer(self, **kwargs: Any) -> nn.Module:
        from ._internal_layers import _InternalConvolutional1DLayer  # slow import on global level

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
        return _InternalConvolutional1DLayer(
            self._input_size.channel,
            self._output_channel,
            self._kernel_size,
            activation_function,
            self._padding,
            self._stride,
        )
    
    @property
    def input_size(self) -> int:
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
    def output_size(self) -> int:
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
            new_size = math.ceil(
                (self._input_size + self._padding * 2 - self._kernel_size + 1) / (1.0 * self._stride),
            )
            self._output_size = new_size
        return self._output_size

    def _set_input_size(self, input_size: int | ModelImageSize) -> None:
        if isinstance(input_size, ModelImageSize):
            raise TypeError("The input_size of a convolution 1d-layer has to be of type int.")
        self._input_size = input_size
        self._output_size = None

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
        if not isinstance(other, Convolutional1DLayer):
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
