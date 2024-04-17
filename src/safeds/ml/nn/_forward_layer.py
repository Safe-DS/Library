from torch import Tensor
from torch.nn import Module, Linear, Sigmoid, ReLU, Softmax

from safeds.exceptions import ClosedBound, OutOfBoundsError
from safeds.ml.nn._layer import _Layer


class _InternalLayer(Module):
    def __init__(self, input_size: int, output_size: int, activation_function: str):
        super().__init__()
        self._layer = Linear(input_size, output_size)
        match activation_function:
            case "sigmoid":
                self._fn = Sigmoid()
            case "relu":
                self._fn = ReLU()
            case "softmax":
                self._fn = Softmax()
            case _:
                raise ValueError("Unknown Activation Function: " + activation_function)

    def forward(self, x: Tensor) -> Tensor:
        return self._fn(self._layer(x))


class ForwardLayer(_Layer):
    def __init__(self, output_size: int, input_size: int | None = None):
        """
        Create a FNN Layer.

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

    def _get_internal_layer(self, activation_function: str) -> _InternalLayer:
        return _InternalLayer(self._input_size, self._output_size, activation_function)

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
