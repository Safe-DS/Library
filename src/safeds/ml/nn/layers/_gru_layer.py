from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _ClosedBound
from safeds.ml.hyperparameters import Choice
from safeds.ml.nn.typing import ModelImageSize

from ._layer import Layer

if TYPE_CHECKING:
    from torch import nn


class GRULayer(Layer):
    """
    A gated recurrent unit (GRU) layer.

    Parameters
    ----------
    neuron_count:
        The number of neurons in this layer

    Raises
    ------
    OutOfBoundsError
        If input_size < 1
        If output_size < 1
    """

    def __init__(self, neuron_count: int | Choice[int]):
        if isinstance(neuron_count, Choice):
            for val in neuron_count:
                _check_bounds("neuron_count", val, lower_bound=_ClosedBound(1))
        else:
            _check_bounds("neuron_count", neuron_count, lower_bound=_ClosedBound(1))

        self._input_size: int | None = None
        self._output_size = neuron_count

    def _get_internal_layer(self, **kwargs: Any) -> nn.Module:
        assert not self._contains_choices()
        assert not isinstance(self._output_size, Choice)  # just for linter
        from ._internal_layers import _InternalGRULayer  # Slow import on global level

        if "activation_function" not in kwargs:
            raise ValueError(
                "The activation_function is not set. The internal layer can only be created when the activation_function is provided in the kwargs.",
            )
        else:
            activation_function: str = kwargs["activation_function"]

        if self._input_size is None:
            raise ValueError("The input_size is not yet set.")

        return _InternalGRULayer(self._input_size, self._output_size, activation_function)

    @property
    def input_size(self) -> int:
        """
        Get the input_size of this layer.

        Returns
        -------
        result:
            The amount of values being passed into this layer.
        """
        if self._input_size is None:
            raise ValueError("The input_size is not yet set.")
        return self._input_size

    @property
    def output_size(self) -> int | Choice[int]:
        """
        Get the output_size of this layer.

        Returns
        -------
        result:
            The number of neurons in this layer.
        """
        return self._output_size

    def _set_input_size(self, input_size: int | ModelImageSize) -> None:
        if isinstance(input_size, ModelImageSize):
            raise TypeError("The input_size of a gru layer has to be of type int.")

        self._input_size = input_size

    def _contains_choices(self) -> bool:
        return isinstance(self._output_size, Choice)

    def _get_layers_for_all_choices(self) -> list[GRULayer]:
        assert self._contains_choices()
        assert isinstance(self._output_size, Choice)  # just for linter
        layers = []
        for val in self._output_size:
            layers.append(GRULayer(neuron_count=val))
        return layers

    def __hash__(self) -> int:
        return _structural_hash(
            self._input_size,
            self._output_size,
        )  # pragma: no cover

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GRULayer):
            return NotImplemented
        return (self is other) or (self._input_size == other._input_size and self._output_size == other._output_size)

    def __sizeof__(self) -> int:
        return sys.getsizeof(self._input_size) + sys.getsizeof(self._output_size)
