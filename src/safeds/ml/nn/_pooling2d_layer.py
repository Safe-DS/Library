from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

from safeds.data.image.typing import ImageSize

if TYPE_CHECKING:
    from torch import Tensor, nn

from safeds.ml.nn._layer import _Layer


def _create_internal_model(strategy: Literal["max", "avg"], kernel_size: int, padding: int, stride: int) -> nn.Module:
    from torch import nn

    class _InternalLayer(nn.Module):
        def __init__(self, strategy: Literal["max", "avg"], kernel_size: int, padding: int, stride: int):
            super().__init__()
            match strategy:
                case "max":
                    self._layer = nn.MaxPool2d(kernel_size=kernel_size, padding=padding, stride=stride)
                case "avg":
                    self._layer = nn.AvgPool2d(kernel_size=kernel_size, padding=padding, stride=stride)

        def forward(self, x: Tensor) -> Tensor:
            return self._layer(x)

    return _InternalLayer(strategy, kernel_size, padding, stride)


class _Pooling2DLayer(_Layer):
    def __init__(self, strategy: Literal["max", "avg"], kernel_size: int, *, stride: int = -1, padding: int = 0):
        """
        Create a Pooling 2D Layer.

        Parameters
        ----------
        strategy:
            the strategy of the pooling
        kernel_size:
            the size of the kernel
        stride:
            the stride of the pooling
        padding:
            the padding of the pooling
        """
        self._strategy = strategy
        self._kernel_size = kernel_size
        self._stride = stride if stride != -1 else kernel_size
        self._padding = padding
        self._input_size: ImageSize | None = None
        self._output_size: ImageSize | None = None

    def _get_internal_layer(self, **kwargs: Any) -> nn.Module:  # noqa: ARG002
        return _create_internal_model(self._strategy, self._kernel_size, self._padding, self._stride)

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
                (self.input_size.width + self._padding * 2 - self._kernel_size + 1) / (1.0 * self._stride),
            )
            new_height = math.ceil(
                (self.input_size.height + self._padding * 2 - self._kernel_size + 1) / (1.0 * self._stride),
            )
            self._output_size = ImageSize(new_width, new_height, self._input_size.channel, _ignore_invalid_channel=True)
        return self._output_size

    def _set_input_size(self, input_size: int | ImageSize) -> None:
        if isinstance(input_size, int):
            raise TypeError("The input_size of a pooling layer has to be of type ImageSize.")
        self._input_size = input_size
        self._output_size = None


class MaxPooling2DLayer(_Pooling2DLayer):

    def __init__(self, kernel_size: int, *, stride: int = -1, padding: int = 0) -> None:
        """
        Create a maximum Pooling 2D Layer.

        Parameters
        ----------
        kernel_size:
            the size of the kernel
        stride:
            the stride of the pooling
        padding:
            the padding of the pooling
        """
        super().__init__("max", kernel_size, stride=stride, padding=padding)


class AvgPooling2DLayer(_Pooling2DLayer):

    def __init__(self, kernel_size: int, *, stride: int = -1, padding: int = 0) -> None:
        """
        Create a average Pooling 2D Layer.

        Parameters
        ----------
        kernel_size:
            the size of the kernel
        stride:
            the stride of the pooling
        padding:
            the padding of the pooling
        """
        super().__init__("avg", kernel_size, stride=stride, padding=padding)
