from __future__ import annotations

from typing import TYPE_CHECKING, Any

from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _OpenBound
from safeds.ml.nn.typing import ModelImageSize

from ._layer import Layer

if TYPE_CHECKING:
    from torch import nn

class DropoutLayer(Layer):
    """
    Create a dropout Layer.

    Parameters
    ----------
    propability:
        The propability of which the input neuron becomes an output neuron

    Raises
    ------
    OutOfBoundsError
        If propability < 0
        If propability > 1
    """

    def __init__(self, propability: float):
        _check_bounds("propability", propability, lower_bound=_OpenBound(0), upper_bound=_OpenBound(1))
        self.propability = propability
        self._input_size: int | None = None

    def _get_internal_layer(self, **kwargs: Any) -> nn.Module:
        from ._internal_layers import _InternalDropoutLayer  # slow import on global level

        if self._input_size is None:
            raise ValueError(
                "The input_size is not yet set. The internal layer can only be created when the input_size is set.",
            )
        if self._output_size is None:
            raise ValueError(
                "The output_size is not yet set. The internal layer can only be created when the output_size is set.",
            )
        return _InternalDropoutLayer(self.propability)
    
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
            The amount of values being passed out of this layer.

        Raises
        ------
        ValueError
            If the input_size is not yet set
        """
        if self._input_size is None:
            raise ValueError("The input_size is not yet set.")
        return self._input_size
    
    def _set_input_size(self, input_size: int) -> None:
        self._input_size = input_size

    def __hash__(self) -> int:
        return _structural_hash(self._input_size)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DropoutLayer):
            return NotImplemented
        return (self is other) or (self._input_size == other._input_size)

    def __sizeof__(self) -> int:
        return self._input_size