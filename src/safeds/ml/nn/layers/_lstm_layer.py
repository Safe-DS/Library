from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from safeds._config import _init_default_device
from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _ClosedBound
from safeds.ml.nn.typing import ModelImageSize

from ._layer import Layer

if TYPE_CHECKING:
    from torch import Tensor, nn


class LSTMLayer(Layer):
    """
    A long short-term memory (LSTM) layer.

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

    def __init__(self, neuron_count: int):
        _check_bounds("neuron_count", neuron_count, lower_bound=_ClosedBound(1))

        self._input_size: int | None = None
        self._output_size = neuron_count

    def _get_internal_layer(self, **kwargs: Any) -> nn.Module:
        if "activation_function" not in kwargs:
            raise ValueError(
                "The activation_function is not set. The internal layer can only be created when the activation_function is provided in the kwargs.",
            )
        else:
            activation_function: str = kwargs["activation_function"]

        if self._input_size is None:
            raise ValueError("The input_size is not yet set.")

        return _create_internal_model(self._input_size, self._output_size, activation_function)

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
    def output_size(self) -> int:
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
            raise TypeError("The input_size of a forward layer has to be of type int.")

        self._input_size = input_size

    def __hash__(self) -> int:
        return _structural_hash(
            self._input_size,
            self._output_size,
        )  # pragma: no cover

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LSTMLayer):
            return NotImplemented
        return (self is other) or (self._input_size == other._input_size and self._output_size == other._output_size)

    def __sizeof__(self) -> int:
        return sys.getsizeof(self._input_size) + sys.getsizeof(self._output_size)


def _create_internal_model(input_size: int, output_size: int, activation_function: str) -> nn.Module:
    from torch import nn

    _init_default_device()

    class _InternalLayer(nn.Module):
        def __init__(self, input_size: int, output_size: int, activation_function: str):
            super().__init__()
            self._layer = nn.LSTM(input_size, output_size)
            match activation_function:
                case "sigmoid":
                    self._fn = nn.Sigmoid()
                case "relu":
                    self._fn = nn.ReLU()
                case "softmax":
                    self._fn = nn.Softmax()
                case "none":
                    self._fn = None
                case _:
                    raise ValueError("Unknown Activation Function: " + activation_function)

        def forward(self, x: Tensor) -> Tensor:
            return self._fn(self._layer(x)[0]) if self._fn is not None else self._layer(x)[0]

    return _InternalLayer(input_size, output_size, activation_function)
