import torch.nn as nn


class fnn_layer():
    def __init__(self, input_size, output_size):
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
