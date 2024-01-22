import numpy as np
import torch
from torch import nn

from safeds.data.tabular.containers import TaggedTable
from safeds.ml.nn._fnn_layer import FNNLayer


class RegressionModel:
    def __init__(self, layers: list):
        self._model = PytorchModel(layers, is_for_classification=False)
        self._batch_size = 1

    def train(self, train_data: TaggedTable, epoch_size: int = 25, batch_size: int = 1) -> None:
        """
        Train the neural network with given training data.

        Parameters
        ----------
        train_data : TaggedTable
            The data the network should be trained on.
        epoch_size : int
            The number of times the training cycle should be done
        batch_size : int
            The size of data batches that should be loaded at one time.

        Raises
        ------
        ValueError
            If epoch_size < 1
            If batch_size < 1

        """
        if epoch_size < 1:
            raise ValueError("The Number of Epochs must be at least 1")
        if batch_size < 1:
            raise ValueError("Batch Size must be at least 1")
        self._batch_size = batch_size
        dataloader = train_data.into_dataloader(self._batch_size)

        loss_fn = nn.MSELoss()

        optimizer = torch.optim.SGD(self._model.parameters(), lr=0.05)

        loss_values = []
        accuracies = []
        for _epoch in range(epoch_size):
            # print(f"Epoch {epoch+1}")
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
        # print(loss_values)

    def predict(self, test_data: TaggedTable) -> None:
        """
        Make a prediction for the given test data.

        Parameters
        ----------
        test_data : TaggedTable
            The data the network should try to predict.

        """
        dataloader = test_data.into_dataloader(self._batch_size)
        self._model.eval()
        loss_values_test = []
        accuracies_test = []
        loss_fn = nn.MSELoss()
        with torch.no_grad():
            for x, y in dataloader:
                pred = self._model(x)
                loss = loss_fn(pred, y)
                loss_values_test.append(loss.item())
                accuracy = torch.mean(1 - torch.abs(pred - y))
                accuracies_test.append(accuracy.item())

        # print(np.mean(loss_values_test))
        # print(np.mean(accuracies_test))


class ClassificationModel:
    def __init__(self, layers: list):
        self._model = PytorchModel(layers, is_for_classification=True)
        self._batch_size = 1

    def train(self, train_data: TaggedTable, epoch_size: int = 25, batch_size: int = 1) -> None:
        """
        Train the neural network with given training data.

        Parameters
        ----------
        train_data : TaggedTable
            The data the network should be trained on.
        epoch_size : int
            The number of times the training cycle should be done
        batch_size : int
            The size of data batches that should be loaded at one time.

        Raises
        ------
        ValueError
            If epoch_size < 1
            If batch_size < 1

        """
        if epoch_size < 1:
            raise ValueError("The Number of Epochs must be at least 1")
        if batch_size < 1:
            raise ValueError("Batch Size must be at least 1")
        self._batch_size = batch_size
        dataloader = train_data.into_dataloader(self._batch_size)

        loss_fn = nn.BCELoss()

        optimizer = torch.optim.SGD(self._model.parameters(), lr=0.05)

        loss_values = []
        accuracies = []
        for _epoch in range(epoch_size):
            # print(f"Epoch {epoch+1}")
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
        # print(loss_values)

    def predict(self, test_data: TaggedTable) -> None:
        """
        Make a prediction for the given test data.

        Parameters
        ----------
        test_data : TaggedTable
            The data the network should try to predict.

        """
        dataloader = test_data.into_dataloader(self._batch_size)
        self._model.eval()
        loss_values_test = []
        accuracies_test = []
        loss_fn = nn.MSELoss()
        with torch.no_grad():
            for x, y in dataloader:
                pred = self._model(x)
                loss = loss_fn(pred, y)
                loss_values_test.append(loss.item())
                accuracy = torch.mean(1 - torch.abs(pred - y))
                accuracies_test.append(accuracy.item())

        # print(np.mean(loss_values_test))
        # print(np.mean(accuracies_test))

    def is_for_regression(self) -> bool:
        return self._model.last_layer_has_output_size_one()


class PytorchModel(nn.Module):
    def __init__(self, layer_list: list[FNNLayer], is_for_classification: bool) -> None:
        super().__init__()
        self._layer_list = layer_list
        layers = []
        for layer in layer_list:
            layers.append(layer.get_pytorch_layer(is_for_classification))

        self._pytorch_layers = nn.ModuleList(layers)

    def forward(self, x: float) -> float:
        for layer in self._pytorch_layers:
            x = layer(x)
        return x
