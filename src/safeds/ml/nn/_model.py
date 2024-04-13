import copy
from collections.abc import Callable
from typing import Self

import torch
from torch import Tensor, nn

from safeds.data.tabular.containers import Column, Table, TaggedTable
from safeds.exceptions import (
    ClosedBound,
    InputSizeError,
    ModelNotFittedError,
    OutOfBoundsError,
    TestTrainDataMismatchError,
)
from safeds.ml.nn._layer import Layer


class NeuralNetworkRegressor:
    def __init__(self, layers: list[Layer]):
        self._model = _InternalModel(layers, is_for_classification=False)
        self._input_size = self._model.input_size
        self._batch_size = 1
        self._is_fitted = False
        self._feature_names: None | list[str] = None
        self._total_number_of_batches_done = 0
        self._total_number_of_epochs_done = 0
        self._loss_sum = 0.0

    def fit(
        self,
        train_data: TaggedTable,
        epoch_size: int = 25,
        batch_size: int = 1,
        learning_rate: float = 0.001,
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
        learning_rate
            The learning rate of the neural network.
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
        if train_data.features.number_of_columns is not self._input_size:
            raise InputSizeError(train_data.features.number_of_columns, self._input_size)
        self._feature_names = train_data.features.column_names
        self._batch_size = batch_size
        copied_model = copy.deepcopy(self)

        dataloader = train_data._into_dataloader(copied_model._batch_size)

        loss_fn = nn.MSELoss()

        optimizer = torch.optim.SGD(copied_model._model.parameters(), lr=learning_rate)
        for _ in range(epoch_size):
            for x, y in iter(dataloader):
                optimizer.zero_grad()

                pred = copied_model._model(x)

                loss = loss_fn(pred, y)
                self._loss_sum += loss.item()
                loss.backward()
                optimizer.step()
                self._total_number_of_batches_done += 1
                if callback_on_batch_completion is not None:
                    callback_on_batch_completion(
                        self._total_number_of_batches_done,
                        self._loss_sum / (self._total_number_of_batches_done * batch_size),
                    )
            self._total_number_of_epochs_done += 1
            if callback_on_epoch_completion is not None:
                callback_on_epoch_completion(
                    self._total_number_of_epochs_done,
                    self._loss_sum / (self._total_number_of_batches_done * batch_size),
                )
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
        if not (sorted(test_data.column_names)).__eq__(
            sorted(self._feature_names) if self._feature_names is not None else None,
        ):
            raise TestTrainDataMismatchError
        dataloader = test_data._into_dataloader(self._batch_size)
        predictions = []
        with torch.no_grad():
            for x in dataloader:
                elem = self._model(x)
                for item in range(len(elem)):
                    predictions.append(elem[item].item())
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


class NeuralNetworkClassifier:
    def __init__(self, layers: list[Layer]):
        self._model = _InternalModel(layers, is_for_classification=True)
        self._input_size = self._model.input_size
        self._batch_size = 1
        self._is_fitted = False
        self._is_multi_class = layers[-1].output_size > 1
        self._feature_names: None | list[str] = None
        self._total_number_of_batches_done = 0
        self._total_number_of_epochs_done = 0
        self._loss_sum = 0.0

    def fit(
        self,
        train_data: TaggedTable,
        epoch_size: int = 25,
        batch_size: int = 1,
        learning_rate: float = 0.001,
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
        learning_rate
            The learning rate of the neural network.
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
        if train_data.features.number_of_columns is not self._input_size:
            raise InputSizeError(train_data.features.number_of_columns, self._input_size)
        self._feature_names = train_data.features.column_names
        self._batch_size = batch_size
        copied_model = copy.deepcopy(self)

        dataloader = train_data._into_dataloader(copied_model._batch_size)

        if self._is_multi_class:
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = nn.BCELoss()

        optimizer = torch.optim.SGD(copied_model._model.parameters(), lr=learning_rate)
        for _ in range(epoch_size):
            for x, y in iter(dataloader):
                optimizer.zero_grad()
                pred = copied_model._model(x)
                if self._is_multi_class:
                    pred_size = Tensor.size(pred, dim=1)
                    predictions_for_all_items_of_batch = []
                    for value in range(len(y)):
                        list_of_probabilities_for_each_category = []
                        class_index = y[value].item()
                        for index in range(pred_size):
                            if index is int(class_index):
                                list_of_probabilities_for_each_category.append(1.0)
                            else:
                                list_of_probabilities_for_each_category.append(0.0)
                        predictions_for_all_items_of_batch.append(list_of_probabilities_for_each_category.copy())

                    y_reshaped_as_tensor_to_fit_format_of_pred = torch.tensor(predictions_for_all_items_of_batch)

                    loss = loss_fn(pred, y_reshaped_as_tensor_to_fit_format_of_pred)
                else:
                    loss = loss_fn(pred, y)
                self._loss_sum += loss.item()
                loss.backward()
                optimizer.step()
                self._total_number_of_batches_done += 1
                if callback_on_batch_completion is not None:
                    callback_on_batch_completion(
                        self._total_number_of_batches_done,
                        self._loss_sum / (self._total_number_of_batches_done * batch_size),
                    )
            if callback_on_epoch_completion is not None:
                callback_on_epoch_completion(
                    self._total_number_of_epochs_done + 1,
                    self._loss_sum / (self._total_number_of_batches_done * batch_size),
                )
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
        if not (sorted(test_data.column_names)).__eq__(
            sorted(self._feature_names) if self._feature_names is not None else None,
        ):
            raise TestTrainDataMismatchError
        dataloader = test_data._into_dataloader(self._batch_size)
        predictions = []
        with torch.no_grad():
            for x in dataloader:
                elem = self._model(x)
                for item in range(len(elem)):
                    if not self._is_multi_class:
                        if elem[item].item() < 0.5:
                            predicted_class = 0  # pragma: no cover
                        else:  # pragma: no cover
                            predicted_class = 1  # pragma: no cover
                        predictions.append(predicted_class)
                    else:
                        values = elem[item].tolist()
                        highest_value = 0
                        category_of_highest_value = 0
                        for index in range(len(values)):
                            if values[index] > highest_value:
                                highest_value = values[index]
                                category_of_highest_value = index
                        predictions.append(category_of_highest_value)
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


class _InternalModel(nn.Module):
    def __init__(self, layers: list[Layer], is_for_classification: bool) -> None:
        super().__init__()
        self._layer_list = layers
        internal_layers = []
        previous_output_size = None

        for layer in layers:
            if previous_output_size is not None:
                layer._set_input_size(previous_output_size)
            internal_layers.append(layer._get_internal_layer(activation_function="relu"))
            previous_output_size = layer.output_size

        if is_for_classification:
            internal_layers.pop()
            if layers[-1].output_size > 2:
                internal_layers.append(layers[-1]._get_internal_layer(activation_function="softmax"))
            else:
                internal_layers.append(layers[-1]._get_internal_layer(activation_function="sigmoid"))
        self._pytorch_layers = nn.Sequential(*internal_layers)

    @property
    def input_size(self) -> int:
        return self._layer_list[0].input_size

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._pytorch_layers:
            x = layer(x)
        return x
