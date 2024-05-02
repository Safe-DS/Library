from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Generic, Self, TypeVar

from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table, TimeSeries
from safeds.exceptions import (
    ClosedBound,
    FeatureDataMismatchError,
    InputSizeError,
    ModelNotFittedError,
    OutOfBoundsError,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor, nn

    from safeds.ml.nn._input_conversion import InputConversion
    from safeds.ml.nn._layer import Layer
    from safeds.ml.nn._output_conversion import OutputConversion

IFT = TypeVar("IFT", TabularDataset, TimeSeries)  # InputFitType
IPT = TypeVar("IPT", Table, TimeSeries)  # InputPredictType
OT = TypeVar("OT", TabularDataset, TimeSeries)  # OutputType


class NeuralNetworkRegressor(Generic[IFT, IPT, OT]):
    def __init__(
        self,
        input_conversion: InputConversion[IFT, IPT],
        layers: list[Layer],
        output_conversion: OutputConversion[IPT, OT],
    ):
        self._input_conversion: InputConversion[IFT, IPT] = input_conversion
        self._model = _create_internal_model(layers, is_for_classification=False)
        self._output_conversion: OutputConversion[IPT, OT] = output_conversion
        self._input_size = self._model.input_size
        self._batch_size = 1
        self._is_fitted = False
        self._total_number_of_batches_done = 0
        self._total_number_of_epochs_done = 0

    def fit(
        self,
        train_data: IFT,
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
        train_data:
            The data the network should be trained on.
        epoch_size:
            The number of times the training cycle should be done.
        batch_size:
            The size of data batches that should be loaded at one time.
        learning_rate:
            The learning rate of the neural network.
        callback_on_batch_completion:
            Function used to view metrics while training. Gets called after a batch is completed with the index of the last batch and the overall loss average.
        callback_on_epoch_completion:
            Function used to view metrics while training. Gets called after an epoch is completed with the index of the last epoch and the overall loss average.

        Returns
        -------
        trained_model:
            The trained Model

        Raises
        ------
        ValueError
            If epoch_size < 1
            If batch_size < 1
        """
        import torch
        from torch import nn

        if epoch_size < 1:
            raise OutOfBoundsError(actual=epoch_size, name="epoch_size", lower_bound=ClosedBound(1))
        if batch_size < 1:
            raise OutOfBoundsError(actual=batch_size, name="batch_size", lower_bound=ClosedBound(1))
        if self._input_conversion._data_size is not self._input_size:
            raise InputSizeError(self._input_conversion._data_size, self._input_size)
        if not self._input_conversion._is_fit_data_valid(train_data):
            raise FeatureDataMismatchError

        copied_model = copy.deepcopy(self)

        copied_model._batch_size = batch_size

        dataloader = copied_model._input_conversion._data_conversion_fit(train_data, copied_model._batch_size)

        loss_fn = nn.MSELoss()

        optimizer = torch.optim.SGD(copied_model._model.parameters(), lr=learning_rate)
        for _ in range(epoch_size):
            loss_sum = 0.0
            amount_of_loss_values_calculated = 0
            for x, y in iter(dataloader):
                optimizer.zero_grad()

                pred = copied_model._model(x)

                loss = loss_fn(pred, y)
                loss_sum += loss.item()
                amount_of_loss_values_calculated += 1
                loss.backward()
                optimizer.step()
                copied_model._total_number_of_batches_done += 1
                if callback_on_batch_completion is not None:
                    callback_on_batch_completion(
                        copied_model._total_number_of_batches_done,
                        loss_sum / amount_of_loss_values_calculated,
                    )
            copied_model._total_number_of_epochs_done += 1
            if callback_on_epoch_completion is not None:
                callback_on_epoch_completion(
                    copied_model._total_number_of_epochs_done,
                    loss_sum / amount_of_loss_values_calculated,
                )
        copied_model._is_fitted = True
        copied_model._model.eval()
        return copied_model

    def predict(self, test_data: IPT) -> OT:
        """
        Make a prediction for the given test data.

        The original Model is not modified.

        Parameters
        ----------
        test_data:
            The data the network should predict.

        Returns
        -------
        prediction:
            The given test_data with an added "prediction" column at the end

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted yet
        """
        import torch

        if not self._is_fitted:
            raise ModelNotFittedError
        if not self._input_conversion._is_predict_data_valid(test_data):
            raise FeatureDataMismatchError
        dataloader = self._input_conversion._data_conversion_predict(test_data, self._batch_size)
        predictions = []
        with torch.no_grad():
            for x in dataloader:
                elem = self._model(x)
                predictions.append(elem.squeeze(dim=1))
        return self._output_conversion._data_conversion(test_data, torch.cat(predictions, dim=0))

    @property
    def is_fitted(self) -> bool:
        """Whether the model is fitted."""
        return self._is_fitted


class NeuralNetworkClassifier(Generic[IFT, IPT, OT]):
    def __init__(
        self,
        input_conversion: InputConversion[IFT, IPT],
        layers: list[Layer],
        output_conversion: OutputConversion[IPT, OT],
    ):
        self._input_conversion: InputConversion[IFT, IPT] = input_conversion
        self._model = _create_internal_model(layers, is_for_classification=True)
        self._output_conversion: OutputConversion[IPT, OT] = output_conversion
        self._input_size = self._model.input_size
        self._batch_size = 1
        self._is_fitted = False
        self._num_of_classes = layers[-1].output_size
        self._total_number_of_batches_done = 0
        self._total_number_of_epochs_done = 0

    def fit(
        self,
        train_data: IFT,
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
        train_data:
            The data the network should be trained on.
        epoch_size:
            The number of times the training cycle should be done.
        batch_size:
            The size of data batches that should be loaded at one time.
        learning_rate:
            The learning rate of the neural network.
        callback_on_batch_completion:
            Function used to view metrics while training. Gets called after a batch is completed with the index of the last batch and the overall loss average.
        callback_on_epoch_completion:
            Function used to view metrics while training. Gets called after an epoch is completed with the index of the last epoch and the overall loss average.

        Returns
        -------
        trained_model:
            The trained Model

        Raises
        ------
        ValueError
            If epoch_size < 1
            If batch_size < 1
        """
        import torch
        from torch import nn

        if epoch_size < 1:
            raise OutOfBoundsError(actual=epoch_size, name="epoch_size", lower_bound=ClosedBound(1))
        if batch_size < 1:
            raise OutOfBoundsError(actual=batch_size, name="batch_size", lower_bound=ClosedBound(1))
        if self._input_conversion._data_size is not self._input_size:
            raise InputSizeError(self._input_conversion._data_size, self._input_size)
        if not self._input_conversion._is_fit_data_valid(train_data):
            raise FeatureDataMismatchError

        copied_model = copy.deepcopy(self)

        copied_model._batch_size = batch_size

        dataloader = copied_model._input_conversion._data_conversion_fit(
            train_data,
            copied_model._batch_size,
            copied_model._num_of_classes,
        )

        if copied_model._num_of_classes > 1:
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = nn.BCELoss()

        optimizer = torch.optim.SGD(copied_model._model.parameters(), lr=learning_rate)
        for _ in range(epoch_size):
            loss_sum = 0.0
            amount_of_loss_values_calculated = 0
            for x, y in iter(dataloader):
                optimizer.zero_grad()
                pred = copied_model._model(x)

                loss = loss_fn(pred, y)
                loss_sum += loss.item()
                amount_of_loss_values_calculated += 1
                loss.backward()
                optimizer.step()

                copied_model._total_number_of_batches_done += 1
                if callback_on_batch_completion is not None:
                    callback_on_batch_completion(
                        copied_model._total_number_of_batches_done,
                        loss_sum / amount_of_loss_values_calculated,
                    )
            copied_model._total_number_of_epochs_done += 1
            if callback_on_epoch_completion is not None:
                callback_on_epoch_completion(
                    copied_model._total_number_of_epochs_done,
                    loss_sum / amount_of_loss_values_calculated,
                )
        copied_model._is_fitted = True
        copied_model._model.eval()
        return copied_model

    def predict(self, test_data: IPT) -> OT:
        """
        Make a prediction for the given test data.

        The original Model is not modified.

        Parameters
        ----------
        test_data:
            The data the network should predict.

        Returns
        -------
        prediction:
            The given test_data with an added "prediction" column at the end

        Raises
        ------
        ModelNotFittedError
            If the Model has not been fitted yet
        """
        import torch

        if not self._is_fitted:
            raise ModelNotFittedError
        if not self._input_conversion._is_predict_data_valid(test_data):
            raise FeatureDataMismatchError
        dataloader = self._input_conversion._data_conversion_predict(test_data, self._batch_size)
        predictions = []
        with torch.no_grad():
            for x in dataloader:
                elem = self._model(x)
                if self._num_of_classes > 1:
                    predictions.append(torch.argmax(elem, dim=1))
                else:
                    predictions.append(elem.squeeze(dim=1).round())
        return self._output_conversion._data_conversion(test_data, torch.cat(predictions, dim=0))

    @property
    def is_fitted(self) -> bool:
        """Whether the model is fitted."""
        return self._is_fitted


def _create_internal_model(layers: list[Layer], is_for_classification: bool) -> nn.Module:
    from torch import nn

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

    return _InternalModel(layers, is_for_classification)
