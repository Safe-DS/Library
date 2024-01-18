import torch.nn as nn


class FNNLayer:
    def __init__(self, input_size, output_size):
        """
        Create a FNN Layer.

        Parameters
        ----------
        epoch_size : int
            The number of times the training cycle should be done
        batch_size : int
            The size of data batches that should be loaded at one time.

        Raises
        ------
        ValueError
            If input_size < 1
            If output_size < 1

        """
        if input_size < 1:
            raise ValueError("Input Size must be at least 1")
        if output_size < 1:
            raise ValueError("Output Size must be at least 1")
        self.input_size = input_size
        self.output_size = output_size

    def _get_pytorch_layer(self):
        return PytorchLayer(self.input_size, self.output_size)

    def get_size(self):
        return self.output_size


class PytorchLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.size = output_size
        self.layer = nn.Linear(input_size, output_size)

        self.fn = nn.ReLU()

    def forward(self, x):
        return self.fn(self.layer(x))

    def get_size(self):
        return self.size
