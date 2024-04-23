from __future__ import annotations

from typing import TYPE_CHECKING

from safeds.data.image.typing import ImageSize

if TYPE_CHECKING:
    from torch import Tensor, nn

from safeds.ml.nn._layer import _Layer


def _create_internal_model() -> nn.Module:
    from torch import nn

    class _InternalLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self._layer = nn.Flatten()

        def forward(self, x: Tensor) -> Tensor:
            return self._layer(x)

    return _InternalLayer()


class FlattenLayer(_Layer):
    def __init__(self):
        """Create a Flatten Layer."""
        self._input_size: ImageSize | None = None
        self._output_size: ImageSize | None = None

    def _get_internal_layer(self) -> nn.Module:
        return _create_internal_model()

    @property
    def input_size(self) -> ImageSize:
        """
        Get the input_size of this layer.

        Returns
        -------
        result :
            The amount of values being passed into this layer.
        """
        return self._input_size

    @property
    def output_size(self) -> int:
        """
        Get the output_size of this layer.

        Returns
        -------
        result :
            The Number of Neurons in this layer.
        """
        return self._input_size.width * self._input_size.height * self._input_size.channel if self._input_size is not None else None

    def _set_input_size(self, input_size: ImageSize) -> None:
        self._input_size = input_size
