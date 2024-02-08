from torch import nn

from safeds.exceptions import ClosedBound, OutOfBoundsError


class _InternalLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int, is_last_layer_of_classification_model: bool):
        super().__init__()
        self._layer = nn.Linear(input_size, output_size)
        if is_last_layer_of_classification_model:
            self._fn = nn.Sigmoid()
        else:
            self._fn = nn.ReLU()

    def forward(self, x: float) -> float:
        return self._fn(self._layer(x))


class FNNLayer:
    def __init__(self, input_size: int, output_size: int):
        """
        Create a FNN Layer.

        Parameters
        ----------
        input_size : int
            The number of neurons in the previous layer
        output_size : int
            The number of neurons in this layer

        Raises
        ------
        ValueError
            If input_size < 1
            If output_size < 1

        """
        if input_size < 1:
            raise OutOfBoundsError(actual=input_size, name="input_size", lower_bound=ClosedBound(1))
        if output_size < 1:
            raise OutOfBoundsError(actual=output_size, name="output_size", lower_bound=ClosedBound(1))
        self._input_size = input_size
        self._output_size = output_size

    def _get_internal_layer(self, is_last_layer_of_classification_model: bool) -> _InternalLayer:
        return _InternalLayer(self._input_size, self._output_size, is_last_layer_of_classification_model)

    @property
    def output_size(self) -> int:
        """
        Get the output_size of this layer.

        Returns
        -------
        result: int
            The Number of Neurons in this layer.
        """
        return self._output_size
