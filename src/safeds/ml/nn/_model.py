import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from safeds.ml.nn import RNN_Layer
from safeds.data.tabular.containers import Column, Table, TaggedTable, TimeSeries
from safeds.exceptions import ColumnSizeError, DuplicateColumnNameError

class Model():
    def __init__(self, layers : list):
        self._model = PyTorchModel(layers) 

    
    def from_layers(self, layers: list):
        pass

    #this is just a demo function
    def model_forward(self, data : DataLoader):
        for batch in iter(data):
            inputs, labels = batch
            inputs = inputs.to(torch.float32)
            self._model(inputs)
    
    
    def train(self,x):
        pass

class PyTorchModel(nn.Module):
    def __init__(self, LayerListe :list[RNN_Layer]):
        super(PyTorchModel, self).__init__()
        self.layerliste = []
        for layer in LayerListe:
            self.layerliste.append(layer._create_pytorch_layer())

    def forward(self, x):
        out = x
        for layer in self.layerliste:
            out = layer(out)
        return out
