from torch import nn

from safeds.exceptions import ClosedBound, OutOfBoundsError


class _PytorchLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int, is_for_classification: bool):
        super().__init__()
        self.size = output_size
        self.layer = nn.Linear(input_size, output_size)
        if is_for_classification:
            self.fn = nn.ReLU()
        else:
            self.fn = nn.Sigmoid()

    def forward(self, x: float) -> float:
        return self.fn(self.layer(x))

    def get_size(self) -> int:
        return self.size


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
        self.input_size = input_size
        self.output_size = output_size

    def get_pytorch_layer(self, is_for_classification: bool) -> _PytorchLayer:
        return _PytorchLayer(self.input_size, self.output_size, is_for_classification)

    def get_size(self) -> int:
        return self.output_size
