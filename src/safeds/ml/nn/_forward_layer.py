from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor, nn

from safeds._utils import _structural_hash
from safeds.exceptions import ClosedBound, OutOfBoundsError
from safeds.ml.nn._layer import _Layer


def _create_internal_model(input_size: int, output_size: int, activation_function: str) -> nn.Module:
    from torch import nn

    class _InternalLayer(nn.Module):
        def __init__(self, input_size: int, output_size: int, activation_function: str):
            super().__init__()
            self._layer = nn.Linear(input_size, output_size)
            match activation_function:
                case "sigmoid":
                    self._fn = nn.Sigmoid()
                case "relu":
                    self._fn = nn.ReLU()
                case "softmax":
                    self._fn = nn.Softmax()
                case _:
                    raise ValueError("Unknown Activation Function: " + activation_function)

        def forward(self, x: Tensor) -> Tensor:
            return self._fn(self._layer(x))

    return _InternalLayer(input_size, output_size, activation_function)

 
class ForwardLayer(_Layer):
    def __init__(self, output_size: int, input_size: int | None = None):
        """
        Create a Feed Forward Layer.

        Parameters
        ----------
        input_size
            The number of neurons in the previous layer
        output_size
            The number of neurons in this layer

        Raises
        ------
        ValueError
            If input_size < 1
            If output_size < 1

        """
        if input_size is not None:
            self._set_input_size(input_size=input_size)
        if output_size < 1:
            raise OutOfBoundsError(actual=output_size, name="output_size", lower_bound=ClosedBound(1))
        self._output_size = output_size

    def _get_internal_layer(self, activation_function: str) -> nn.Module:
        return _create_internal_model(self._input_size, self._output_size, activation_function)

    @property
    def input_size(self) -> int:
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
        return self._output_size

    def _set_input_size(self, input_size: int) -> None:
        if input_size < 1:
            raise OutOfBoundsError(actual=input_size, name="input_size", lower_bound=ClosedBound(1))
        self._input_size = input_size

    def __hash__(self) -> int:
        """
        Return a deterministic hash value for this forward layer.

        Returns
        -------
        hash:
            the hash value
        """
        return _structural_hash(self._input_size, self._output_size)

    def __eq__(self, other: object) -> bool:
        """
        Compare two forward layer instances.

        Returns
        -------
        equals:
            'True' if input and output size are equal, 'False' otherwise.
        """
        if not isinstance(other, ForwardLayer):
            return NotImplemented
        if self is other:
            return True
        return self._input_size == other._input_size and self._output_size == other._output_size

    def __sizeof__(self) -> int:
        import sys

        """
        Return the complete size of this object.

        Returns
        -------
        size:
            Size of this object in bytes.
        """
        return sys.getsizeof(self._input_size) + sys.getsizeof(self._output_size)
