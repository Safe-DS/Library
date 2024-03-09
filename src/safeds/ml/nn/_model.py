import copy
from collections.abc import Callable
from typing import Self

import torch
from torch import Tensor, nn

from safeds.data.tabular.containers import Column, Table, TaggedTable
from safeds.exceptions import ClosedBound, ModelNotFittedError, OutOfBoundsError
from safeds.ml.nn._fnn_layer import FNNLayer


class RegressionNeuralNetwork:
    def __init__(self, layers: list):
        self._model = _PytorchModel(layers, is_for_classification=False)
        self._batch_size = 1
        self._is_fitted = False

    def fit(
        self,
        train_data: TaggedTable,
        epoch_size: int = 25,
        batch_size: int = 1,
        callback_on_batch_completion: Callable[[int, float], None] | None = None,
        callback_on_epoch_completion: Callable[[int, float], None] | None = None,
    ) -> Self:
        """
        Train the neural network with given training data.

        The original model is not modified.

        Parameters
        ----------
        train_data
            The data the network should be trained on.
        epoch_size
            The number of times the training cycle should be done.
        batch_size
            The size of data batches that should be loaded at one time.
        callback_on_batch_completion
            Function used to view metrics while training. Gets called after a batch is completed with the index of the last batch and the overall loss average.
        callback_on_epoch_completion
            Function used to view metrics while training. Gets called after an epoch is completed with the index of the last epoch and the overall loss average.

        Raises
        ------
        ValueError
            If epoch_size < 1
            If batch_size < 1

        Returns
        -------
        trained_model :
            The trained Model
        """
        if epoch_size < 1:
            raise OutOfBoundsError(actual=epoch_size, name="epoch_size", lower_bound=ClosedBound(1))
        if batch_size < 1:
            raise OutOfBoundsError(actual=batch_size, name="batch_size", lower_bound=ClosedBound(1))
        copied_model = copy.deepcopy(self)
        copied_model._batch_size = batch_size
        dataloader = train_data._into_dataloader(copied_model._batch_size)

        loss_fn = nn.MSELoss()

        optimizer = torch.optim.SGD(copied_model._model.parameters(), lr=0.05)
        loss_sum = 0.0
        number_of_batches_done = 0
        for epoch in range(epoch_size):
            for x, y in dataloader:
                optimizer.zero_grad()

                pred = copied_model._model(x)

                loss = loss_fn(pred, y)
                loss_sum += loss
                loss.backward()
                optimizer.step()
                number_of_batches_done += 1
                if callback_on_batch_completion is not None:
                    callback_on_batch_completion(
                        number_of_batches_done, loss_sum / (number_of_batches_done * batch_size),
                    )
            if callback_on_epoch_completion is not None:
                callback_on_epoch_completion(epoch, loss_sum / (number_of_batches_done * batch_size))
        copied_model._is_fitted = True
        copied_model._model.eval()
        return copied_model

    def predict(self, test_data: Table) -> TaggedTable:
        """
        Make a prediction for the given test data.

        The original Model is not modified.

        Parameters
        ----------
        test_data
            The data the network should predict.

        Returns
        -------
        prediction :
            The given test_data with an added "prediction" column at the end

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted yet
        """
        if not self._is_fitted:
            raise ModelNotFittedError
        dataloader = test_data._into_dataloader(self._batch_size)
        predictions = []
        with torch.no_grad():
            for x in dataloader:
                predictions.append(self._model(x).item())
        return test_data.add_column(Column("prediction", predictions)).tag_columns("prediction")

    @property
    def is_fitted(self) -> bool:
        """
        Check if the model is fitted.

        Returns
        -------
        is_fitted
            Whether the model is fitted.
        """
        return self._is_fitted


class ClassificationNeuralNetwork:
    def __init__(self, layers: list[FNNLayer]):
        self._model = _PytorchModel(layers, is_for_classification=True)
        self._batch_size = 1
        self._is_fitted = False
        self._is_multi_class = layers[-1].output_size > 2

    def fit(
        self,
        train_data: TaggedTable,
        epoch_size: int = 25,
        batch_size: int = 1,
        callback_on_batch_completion: Callable[[int, float], None] | None = None,
        callback_on_epoch_completion: Callable[[int, float], None] | None = None,
    ) -> Self:
        """
        Train the neural network with given training data.

        The original model is not modified.

        Parameters
        ----------
        train_data
            The data the network should be trained on.
        epoch_size
            The number of times the training cycle should be done.
        batch_size
            The size of data batches that should be loaded at one time.
        callback_on_batch_completion
            Function used to view metrics while training. Gets called after a batch is completed with the index of the last batch and the overall loss average.
        callback_on_epoch_completion
            Function used to view metrics while training. Gets called after an epoch is completed with the index of the last epoch and the overall loss average.

        Raises
        ------
        ValueError
            If epoch_size < 1
            If batch_size < 1

        Returns
        -------
        trained_model :
            The trained Model
        """
        if epoch_size < 1:
            raise OutOfBoundsError(actual=epoch_size, name="epoch_size", lower_bound=ClosedBound(1))
        if batch_size < 1:
            raise OutOfBoundsError(actual=batch_size, name="batch_size", lower_bound=ClosedBound(1))
        copied_model = copy.deepcopy(self)
        copied_model._batch_size = batch_size
        dataloader = train_data._into_dataloader(copied_model._batch_size)

        if self._is_multi_class:
            loss_fn = nn.CrossEntropyLoss
        else:
            loss_fn = nn.BCELoss()

        optimizer = torch.optim.SGD(copied_model._model.parameters(), lr=0.05)
        loss_sum = 0.0
        number_of_batches_done = 0
        for epoch in range(epoch_size):
            for x, y in dataloader:
                optimizer.zero_grad()
                pred = copied_model._model(x)
                pred_size = Tensor.size(pred, dim=1)
                y_as_list = []
                class_index = y.item()
                for index in range(pred_size):
                    if index is int(class_index):
                        y_as_list.append(1.0)
                    else:
                        y_as_list.append(0.0)

                new_list = [y_as_list]
                y_reshaped_as_tensor = torch.tensor(new_list)

                loss = loss_fn(pred, y_reshaped_as_tensor)
                loss_sum += loss
                loss.backward()
                optimizer.step()
                number_of_batches_done += 1
                if callback_on_batch_completion is not None:
                    callback_on_batch_completion(
                        number_of_batches_done, loss_sum / (number_of_batches_done * batch_size),
                    )
            if callback_on_epoch_completion is not None:
                callback_on_epoch_completion(epoch, loss_sum / (number_of_batches_done * batch_size))
        copied_model._is_fitted = True
        copied_model._model.eval()
        return copied_model

    def predict(self, test_data: Table) -> TaggedTable:
        """
        Make a prediction for the given test data.

        The original Model is not modified.

        Parameters
        ----------
        test_data
            The data the network should predict.

        Returns
        -------
        prediction :
            The given test_data with an added "prediction" column at the end

        Raises
        ------
        ModelNotFittedError
            If the Model has not been fitted yet
        """
        if not self._is_fitted:
            raise ModelNotFittedError
        dataloader = test_data._into_dataloader(self._batch_size)
        predictions = []
        with torch.no_grad():
            for x in dataloader:
                predictions.append(self._model(x).item())
        return test_data.add_column(Column("prediction", predictions)).tag_columns("prediction")

    @property
    def is_fitted(self) -> bool:
        """
        Check if the model is fitted.

        Returns
        -------
        is_fitted :
            Whether the model is fitted.
        """
        return self._is_fitted


class _PytorchModel(nn.Module):
    def __init__(self, fnn_layers: list[FNNLayer], is_for_classification: bool) -> None:
        super().__init__()
        self._layer_list = fnn_layers
        internal_layers = []
        previous_output_size = None

        for layer in fnn_layers:
            if previous_output_size is not None:
                layer._set_input_size(previous_output_size)
            internal_layers.append(layer._get_internal_layer(activation_function="relu"))
            previous_output_size = layer.output_size

        if is_for_classification:
            internal_layers.pop()
            if fnn_layers[-1].output_size > 1:
                internal_layers.append(fnn_layers[-1]._get_internal_layer(activation_function="softmax"))
            else:
                internal_layers.append(fnn_layers[-1]._get_internal_layer(activation_function="sigmoid"))
        self._pytorch_layers = nn.ModuleList(internal_layers)

    def forward(self, x: float) -> float:
        for layer in self._pytorch_layers:
            x = layer(x)
        return x
