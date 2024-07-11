from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from safeds._utils import _structural_hash
from safeds.ml.nn.typing import ConstantImageSize

from ._layer import Layer

if TYPE_CHECKING:
    from torch import nn

    from safeds.ml.nn.typing import ModelImageSize


class FlattenLayer(Layer):
    """A flatten layer."""

    def __init__(self) -> None:
        self._input_size: ModelImageSize | None = None
        self._output_size: int | None = None

    def _get_internal_layer(self, **kwargs: Any) -> nn.Module:  # noqa: ARG002
        from ._internal_layers import _InternalFlattenLayer  # Slow import on global level

        return _InternalFlattenLayer()

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
            self._output_size = self._input_size.width * self._input_size.height * self._input_size.channel
        return self._output_size

    def _set_input_size(self, input_size: int | ModelImageSize) -> None:
        if isinstance(input_size, int):
            raise TypeError("The input_size of a flatten layer has to be of type ImageSize.")
        if not isinstance(input_size, ConstantImageSize):
            raise TypeError("The input_size of a flatten layer has to be a ConstantImageSize.")
        self._input_size = input_size
        self._output_size = None

    def _contains_choices(self) -> bool:
        return False

    def _get_layers_for_all_choices(self) -> list[FlattenLayer]:
        raise NotImplementedError  # pragma: no cover

    def __hash__(self) -> int:
        return _structural_hash(self._input_size, self._output_size)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FlattenLayer):
            return NotImplemented
        return (self is other) or (self._input_size == other._input_size and self._output_size == other._output_size)

    def __sizeof__(self) -> int:
        return sys.getsizeof(self._input_size) + sys.getsizeof(self._output_size)
