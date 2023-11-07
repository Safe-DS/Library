import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from safeds.data.tabular.containers import Column, Table, TaggedTable, TimeSeries
from safeds.exceptions import ColumnSizeError, DuplicateColumnNameError


class RNN_Layer():
    def __init__(self, input_dim, output_dim)-> None:
        self._input_dim = input_dim
        self._output_dim = output_dim


    def _create_pytorch_layer(self):
        return LSTMLayer(self._input_dim, self._output_dim)
    


#definiere LSTM Layer in PyTorch
class LSTMLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LSTMLayer, self).__init__()
        self.lstm = nn.LSTM(input_dim, output_dim, batch_first = True)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return lstm_out