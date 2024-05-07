from __future__ import annotations

from typing import TYPE_CHECKING, Any

from safeds._config import _init_default_device
from safeds.data.image.typing import ImageSize

if TYPE_CHECKING:
    from torch import Tensor, nn

from safeds._utils import _structural_hash
from safeds.exceptions import ClosedBound, OutOfBoundsError
from safeds.ml.nn import Layer


def _create_internal_model(input_size: int, output_size: int, activation_function: str) -> nn.Module:
    from torch import nn

    _init_default_device()

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
                case "none":
                    self._fn = None
                case _:
                    raise ValueError("Unknown Activation Function: " + activation_function)

        def forward(self, x: Tensor) -> Tensor:
            return self._fn(self._layer(x)) if self._fn is not None else self._layer(x)

    return _InternalLayer(input_size, output_size, activation_function)


class ForwardLayer(Layer):
    def __init__(self, output_size: int, input_size: int | None = None):
        """
        Create a Feed Forward Layer.

        Parameters
        ----------
        input_size:
            The number of neurons in the previous layer
        output_size:
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

    def _get_internal_layer(self, **kwargs: Any) -> nn.Module:
        if "activation_function" not in kwargs:
            raise ValueError(
                "The activation_function is not set. The internal layer can only be created when the activation_function is provided in the kwargs.",
            )
        else:
            activation_function: str = kwargs["activation_function"]
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
        return self._input_size

    @property
    def output_size(self) -> int:
        """
        Get the output_size of this layer.

        Returns
        -------
        result:
            The Number of Neurons in this layer.
        """
        return self._output_size

    def _set_input_size(self, input_size: int | ImageSize) -> None:
        if isinstance(input_size, ImageSize):
            raise TypeError("The input_size of a forward layer has to be of type int.")
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
        """
        Return the complete size of this object.

        Returns
        -------
        size:
            Size of this object in bytes.
        """
        import sys

        return sys.getsizeof(self._input_size) + sys.getsizeof(self._output_size)
