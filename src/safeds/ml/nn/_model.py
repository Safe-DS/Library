import pandas as pd
import numpy as np
import torch
import time
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
    
    
    def train(self, train_loader: DataLoader, epochs: int, learningrate : float):
        start_time = time.time()
        criterion = nn.MSELoss()
        optimizer =  torch.optim.Adam(self._model.parameters(), lr = learningrate)

        for epoch in range(epochs):
            for batch in iter(train_loader):
                inputs, labels = batch
                optimizer.zero_grad()

                labels = labels.to(torch.float32)
                inputs = inputs.to(torch.float32)
                outputs = self._model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f'Epoch: {epoch+1:2} Loss {loss.item():10.8f}')
        print(f'\nDuration: {time.time()-start_time:.0f} seconds')

class PyTorchModel(nn.Module):
    def __init__(self, LayerListe :list[RNN_Layer]):
        super(PyTorchModel, self).__init__()
        layers = []
        for layer in LayerListe:
            layers.append(layer._create_pytorch_layer())

        self._layerliste = nn.ModuleList(layers)
        
    def forward(self, x):
        out = x
        for layer in self._layerliste:
            out = layer(out)
        return out
