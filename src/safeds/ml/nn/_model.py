import copy
from typing import Self

import numpy as np
import torch
from torch import nn

from safeds.data.tabular.containers import Column, Table, TaggedTable
from safeds.exceptions import ClosedBound, OutOfBoundsError
from safeds.ml.classical.classification import Classifier
from safeds.ml.classical.regression import Regressor
from safeds.ml.nn._fnn_layer import FNNLayer


class RegressionNeuralNetwork(Regressor):
    def __init__(self, layers: list):
        self._model = _PytorchModel(layers, is_for_classification=False)
        self._batch_size = 1

    def fit(self, train_data: TaggedTable, epoch_size: int = 25, batch_size: int = 1) -> Self:
        """
        Train the neural network with given training data.

        The original model is not modified.

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

        Returns
        -------
        RegressionNeuralNetwork
            The trained Model
        """
        if epoch_size < 1:
            raise OutOfBoundsError(actual=epoch_size, name="epoch_size", lower_bound=ClosedBound(1))
        if batch_size < 1:
            raise OutOfBoundsError(actual=batch_size, name="batch_size", lower_bound=ClosedBound(1))
        copied_model = copy.deepcopy(self)
        copied_model._batch_size = batch_size
        dataloader = train_data.into_dataloader(copied_model._batch_size)

        loss_fn = nn.MSELoss()

        optimizer = torch.optim.SGD(copied_model._model.parameters(), lr=0.05)

        loss_values = []
        accuracies = []
        for _epoch in range(epoch_size):
            tmp_loss = []
            tmp_accuracies = []
            for x, y in dataloader:
                pred = copied_model._model(x)

                loss = loss_fn(pred, y)
                tmp_loss.append(loss.item())

                accuracy = torch.mean(1 - torch.abs(pred - y))
                tmp_accuracies.append(accuracy.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_values.append(np.mean(tmp_loss))
            accuracies.append(np.mean(tmp_accuracies))
        return copied_model

    def predict(self, test_data: Table) -> TaggedTable:
        """
        Make a prediction for the given test data.

        The original Model is not modified.

        Parameters
        ----------
        test_data : Table
            The data the network should predict.

        Returns
        -------
        TaggedTable
            The given test_data with an added "prediction" column at the end
        """
        copied_model = copy.deepcopy(self)
        test_data.add_column(Column("predictions"))
        tagged_test_data = test_data.tag_columns("predictions")
        dataloader = tagged_test_data.into_dataloader(copied_model._batch_size)
        copied_model._model.eval()
        predictions = []
        with torch.no_grad():
            for x, _y in dataloader:
                predictions.append(copied_model._model(x))
        tagged_test_data.replace_column("predictions", [Column("prediction", predictions)])
        return tagged_test_data


class ClassificationNeuralNetwork(Classifier):
    def __init__(self, layers: list):
        self._model = _PytorchModel(layers, is_for_classification=True)
        self._batch_size = 1

    def fit(self, train_data: TaggedTable, epoch_size: int = 25, batch_size: int = 1) -> Self:
        """
        Train the neural network with given training data.

        The original model is not modified.

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

        Returns
        -------
        ClassificationNeuralNetwork
            The trained Model
        """
        if epoch_size < 1:
            raise OutOfBoundsError(actual=epoch_size, name="epoch_size", lower_bound=ClosedBound(1))
        if batch_size < 1:
            raise OutOfBoundsError(actual=batch_size, name="batch_size", lower_bound=ClosedBound(1))
        copied_model = copy.deepcopy(self)
        copied_model._batch_size = batch_size
        dataloader = train_data.into_dataloader(copied_model._batch_size)

        loss_fn = nn.BCELoss()

        optimizer = torch.optim.SGD(copied_model._model.parameters(), lr=0.05)

        loss_values = []
        accuracies = []
        for _epoch in range(epoch_size):
            tmp_loss = []
            tmp_accuracies = []
            for x, y in dataloader:
                pred = copied_model._model(x)

                loss = loss_fn(pred, y)
                tmp_loss.append(loss.item())

                accuracy = torch.mean(1 - torch.abs(pred - y))
                tmp_accuracies.append(accuracy.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_values.append(np.mean(tmp_loss))
            accuracies.append(np.mean(tmp_accuracies))
        return copied_model

    def predict(self, test_data: Table) -> TaggedTable:
        """
        Make a prediction for the given test data.

        The original Model is not modified.

        Parameters
        ----------
        test_data : Table
            The data the network should predict.

        Returns
        -------
        TaggedTable
            The given test_data with an added "prediction" column at the end
        """
        copied_model = copy.deepcopy(self)
        test_data.add_column(Column("predictions"))
        tagged_test_data = test_data.tag_columns("predictions")
        dataloader = tagged_test_data.into_dataloader(copied_model._batch_size)
        copied_model._model.eval()
        predictions = []
        with torch.no_grad():
            for x, _y in dataloader:
                predictions.append(copied_model._model(x))
        tagged_test_data.replace_column("predictions", [Column("prediction"), predictions])
        return tagged_test_data


class _PytorchModel(nn.Module):
    def __init__(self, layer_list: list[FNNLayer], is_for_classification: bool) -> None:
        super().__init__()
        self._layer_list = layer_list
        layers = []
        for layer in layer_list:
            layers.append(layer._get_internal_layer(is_for_classification))

        self._pytorch_layers = nn.ModuleList(layers)

    def forward(self, x: float) -> float:
        for layer in self._pytorch_layers:
            x = layer(x)
        return x
