import numpy as np
from torch import nn
import torch
from safeds.data.tabular.containers import TaggedTable
from safeds.ml.nn import fnn_layer


class Model:
    def __init__(self, layers: list):
        self._model = PytorchModel(layers)

    def train(self, train_data: TaggedTable, epoch_size=25, batch_size=1):
        dataloader = train_data.into_dataloader(batch_size)

        if self.is_for_regression():
            loss_fn = nn.MSELoss()
        else:
            loss_fn = nn.BCELoss()

        optimizer = torch.optim.SGD(self._model.parameters(), lr=0.05)

        loss_values = []
        accuracies = []
        num_epochs = epoch_size
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}")
            tmp_loss = []
            tmp_accuracies = []
            for x, y in dataloader:
                pred = self._model(x)

                loss = loss_fn(pred, y)
                tmp_loss.append(loss.item())

                accuracy = torch.mean(1 - torch.abs(pred - y))
                tmp_accuracies.append(accuracy.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_values.append(np.mean(tmp_loss))
            accuracies.append(np.mean(tmp_accuracies))
        print(loss_values)

    def predict(self, test_data: TaggedTable):
        dataloader = test_data.into_dataloader()
        self._model.eval()
        loss_values_test = []
        accuracies_test = []
        loss_fn = nn.MSELoss()
        with torch.no_grad():
            for X, y in dataloader:
                pred = self._model(X)
                loss = loss_fn(pred, y)
                loss_values_test.append(loss.item())
                accuracy = torch.mean(1 - torch.abs(pred - y))
                accuracies_test.append(accuracy.item())

        print(np.mean(loss_values_test))
        print(np.mean(accuracies_test))

    def is_for_regression(self):
        return self._model.last_layer_has_output_size_one()

class PytorchModel(nn.Module):
    def __init__(self, layer_list: list[fnn_layer]):
        super().__init__()
        self._layer_list = layer_list
        layers = []
        for layer in layer_list:
            layers.append(layer._get_pytorch_layer())

        self._pytorch_layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self._pytorch_layers:
            x = layer(x)
        return x

    def last_layer_has_output_size_one(self):
        return self._layer_list[-1].get_size() == 1
