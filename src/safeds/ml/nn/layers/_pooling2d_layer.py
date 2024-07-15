from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING, Any, Literal

from safeds._utils import _structural_hash

from ._layer import Layer

if TYPE_CHECKING:
    from torch import nn

    from safeds.ml.nn.typing import ModelImageSize


class _Pooling2DLayer(Layer):
    """
    A pooling 2D Layer.

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

    def __init__(self, strategy: Literal["max", "avg"], kernel_size: int, *, stride: int = -1, padding: int = 0):
        self._strategy = strategy
        self._kernel_size = kernel_size
        self._stride = stride if stride != -1 else kernel_size
        self._padding = padding
        self._input_size: ModelImageSize | None = None
        self._output_size: ModelImageSize | None = None

    def _get_internal_layer(self, **kwargs: Any) -> nn.Module:  # noqa: ARG002
        from ._internal_layers import _InternalPooling2DLayer  # Slow import on global level

        return _InternalPooling2DLayer(self._strategy, self._kernel_size, self._padding, self._stride)

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
                (self.input_size.width + self._padding * 2 - self._kernel_size + 1) / (1.0 * self._stride),
            )
            new_height = math.ceil(
                (self.input_size.height + self._padding * 2 - self._kernel_size + 1) / (1.0 * self._stride),
            )
            self._output_size = self._input_size.__class__(
                new_width,
                new_height,
                self._input_size.channel,
                _ignore_invalid_channel=True,
            )
        return self._output_size

    def _set_input_size(self, input_size: int | ModelImageSize) -> None:
        if isinstance(input_size, int):
            raise TypeError("The input_size of a pooling layer has to be of type ImageSize.")
        self._input_size = input_size
        self._output_size = None

    def _contains_choices(self) -> bool:
        return False

    def _get_layers_for_all_choices(self) -> list[_Pooling2DLayer]:
        raise NotImplementedError  # pragma: no cover

    def __hash__(self) -> int:
        return _structural_hash(
            self._strategy,
            self._kernel_size,
            self._stride,
            self._padding,
            self._input_size,
            self._output_size,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self is other) or (
            self._input_size == other._input_size
            and self._output_size == other._output_size
            and self._strategy == other._strategy
            and self._kernel_size == other._kernel_size
            and self._stride == other._stride
            and self._padding == other._padding
        )

    def __sizeof__(self) -> int:
        return (
            sys.getsizeof(self._input_size)
            + sys.getsizeof(self._output_size)
            + sys.getsizeof(self._strategy)
            + sys.getsizeof(self._kernel_size)
            + sys.getsizeof(self._stride)
            + sys.getsizeof(self._padding)
        )


class MaxPooling2DLayer(_Pooling2DLayer):
    """
    A maximum Pooling 2D Layer.

    Parameters
    ----------
    kernel_size:
        the size of the kernel
    stride:
        the stride of the pooling
    padding:
        the padding of the pooling
    """

    def __init__(self, kernel_size: int, *, stride: int = -1, padding: int = 0) -> None:
        super().__init__("max", kernel_size, stride=stride, padding=padding)


class AveragePooling2DLayer(_Pooling2DLayer):
    """
    An average pooling 2D Layer.

    Parameters
    ----------
    kernel_size:
        the size of the kernel
    stride:
        the stride of the pooling
    padding:
        the padding of the pooling
    """

    def __init__(self, kernel_size: int, *, stride: int = -1, padding: int = 0) -> None:
        super().__init__("avg", kernel_size, stride=stride, padding=padding)
