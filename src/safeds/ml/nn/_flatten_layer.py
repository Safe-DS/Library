from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from safeds._config import _init_default_device
from safeds._utils import _structural_hash

if TYPE_CHECKING:
    from torch import Tensor, nn

    from safeds.data.image.typing import ImageSize

from safeds.ml.nn import Layer


def _create_internal_model() -> nn.Module:
    from torch import nn

    _init_default_device()

    class _InternalLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._layer = nn.Flatten()

        def forward(self, x: Tensor) -> Tensor:
            return self._layer(x)

    return _InternalLayer()


class FlattenLayer(Layer):
    def __init__(self) -> None:
        """Create a Flatten Layer."""
        self._input_size: ImageSize | None = None
        self._output_size: int | None = None

    def _get_internal_layer(self, **kwargs: Any) -> nn.Module:  # noqa: ARG002
        return _create_internal_model()

    @property
    def input_size(self) -> ImageSize:
        """
        Get the input_size of this layer.

        Returns
        -------
        result :
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
        result :
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
            self._output_size = self._input_size.width * self._input_size.height * self._input_size.channel
        return self._output_size

    def _set_input_size(self, input_size: int | ImageSize) -> None:
        if isinstance(input_size, int):
            raise TypeError("The input_size of a flatten layer has to be of type ImageSize.")
        self._input_size = input_size
        self._output_size = None

    def __hash__(self) -> int:
        """
        Return a deterministic hash value for this flatten layer.

        Returns
        -------
        hash:
            the hash value
        """
        return _structural_hash(self._input_size, self._output_size)

    def __eq__(self, other: object) -> bool:
        """
        Compare two flatten layer.

        Parameters
        ----------
        other:
            The flatten layer to compare to.

        Returns
        -------
        equals:
            Whether the two flatten layer are the same.
        """
        if not isinstance(other, FlattenLayer):
            return NotImplemented
        return (self is other) or (self._input_size == other._input_size and self._output_size == other._output_size)

    def __sizeof__(self) -> int:
        """
        Return the complete size of this object.

        Returns
        -------
        size:
            Size of this object in bytes.
        """
        return sys.getsizeof(self._input_size) + sys.getsizeof(self._output_size)
