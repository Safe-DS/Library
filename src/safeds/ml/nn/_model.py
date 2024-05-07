from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Generic, Self, TypeVar

from safeds._config import _init_default_device
from safeds.data.image.containers import ImageList
from safeds.data.labeled.containers import ImageDataset, TabularDataset, TimeSeriesDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import (
    ClosedBound,
    FeatureDataMismatchError,
    InputSizeError,
    InvalidModelStructureError,
    ModelNotFittedError,
    OutOfBoundsError,
)
from safeds.ml.nn import (
    Convolutional2DLayer,
    FlattenLayer,
    ForwardLayer,
    InputConversionImage,
    OutputConversionImageToColumn,
    OutputConversionImageToImage,
    OutputConversionImageToTable,
)
from safeds.ml.nn._output_conversion_image import _OutputConversionImage
from safeds.ml.nn._pooling2d_layer import _Pooling2DLayer

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor, nn

    from safeds.data.image.typing import ImageSize
    from safeds.ml.nn import InputConversion, Layer, OutputConversion


IFT = TypeVar("IFT", TabularDataset, TimeSeriesDataset, ImageDataset)  # InputFitType
IPT = TypeVar("IPT", Table, TimeSeriesDataset, ImageList)  # InputPredictType
OT = TypeVar("OT", TabularDataset, TimeSeriesDataset, ImageDataset)  # OutputType


class NeuralNetworkRegressor(Generic[IFT, IPT, OT]):
    """
    A NeuralNetworkRegressor is a neural network that is used for regression tasks.

    Parameters
    ----------
    input_conversion:
        to convert the input data for the neural network
    layers:
        a list of layers for the neural network to learn
    output_conversion:
        to convert the output data of the neural network back

    Raises
    ------
    InvalidModelStructureError
        if the defined model structure is invalid
    """

    def __init__(
        self,
        input_conversion: InputConversion[IFT, IPT],
        layers: list[Layer],
        output_conversion: OutputConversion[IPT, OT],
    ):
        if len(layers) == 0:
            raise InvalidModelStructureError("You need to provide at least one layer to a neural network.")
        if isinstance(input_conversion, InputConversionImage):
            if not isinstance(output_conversion, _OutputConversionImage):
                raise InvalidModelStructureError(
                    "The defined model uses an input conversion for images but no output conversion for images.",
                )
            elif isinstance(output_conversion, OutputConversionImageToColumn | OutputConversionImageToTable):
                raise InvalidModelStructureError(
                    "A NeuralNetworkRegressor cannot be used with images as input and 1-dimensional data as output.",
                )
            data_dimensions = 2
            for layer in layers:
                if data_dimensions == 2 and (isinstance(layer, Convolutional2DLayer | _Pooling2DLayer)):
                    continue
                elif data_dimensions == 2 and isinstance(layer, FlattenLayer):
                    data_dimensions = 1
                elif data_dimensions == 1 and isinstance(layer, ForwardLayer):
                    continue
                else:
                    raise InvalidModelStructureError(
                        (
                            "The 2-dimensional data has to be flattened before using a 1-dimensional layer."
                            if data_dimensions == 2
                            else "You cannot use a 2-dimensional layer with 1-dimensional data."
                        ),
                    )
            if data_dimensions == 1 and isinstance(output_conversion, OutputConversionImageToImage):
                raise InvalidModelStructureError(
                    "The output data would be 1-dimensional but the provided output conversion uses 2-dimensional data.",
                )
        elif isinstance(output_conversion, _OutputConversionImage):
            raise InvalidModelStructureError(
                "The defined model uses an output conversion for images but no input conversion for images.",
            )
        else:
            for layer in layers:
                if isinstance(layer, Convolutional2DLayer | FlattenLayer | _Pooling2DLayer):
                    raise InvalidModelStructureError("You cannot use a 2-dimensional layer with 1-dimensional data.")

        self._input_conversion: InputConversion[IFT, IPT] = input_conversion
        self._model = _create_internal_model(input_conversion, layers, is_for_classification=False)
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

        _init_default_device()

        if not self._input_conversion._is_fit_data_valid(train_data):
            raise FeatureDataMismatchError
        if epoch_size < 1:
            raise OutOfBoundsError(actual=epoch_size, name="epoch_size", lower_bound=ClosedBound(1))
        if batch_size < 1:
            raise OutOfBoundsError(actual=batch_size, name="batch_size", lower_bound=ClosedBound(1))
        if self._input_conversion._data_size is not self._input_size:
            raise InputSizeError(self._input_conversion._data_size, self._input_size)

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

        _init_default_device()

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
        return self._output_conversion._data_conversion(
            test_data,
            torch.cat(predictions, dim=0),
            **self._input_conversion._get_output_configuration(),
        )

    @property
    def is_fitted(self) -> bool:
        """Whether the model is fitted."""
        return self._is_fitted


class NeuralNetworkClassifier(Generic[IFT, IPT, OT]):
    """
    A NeuralNetworkClassifier is a neural network that is used for classification tasks.

    Parameters
    ----------
    input_conversion:
        to convert the input data for the neural network
    layers:
        a list of layers for the neural network to learn
    output_conversion:
        to convert the output data of the neural network back

    Raises
    ------
    InvalidModelStructureError
        if the defined model structure is invalid
    """

    def __init__(
        self,
        input_conversion: InputConversion[IFT, IPT],
        layers: list[Layer],
        output_conversion: OutputConversion[IPT, OT],
    ):
        if len(layers) == 0:
            raise InvalidModelStructureError("You need to provide at least one layer to a neural network.")
        if isinstance(output_conversion, OutputConversionImageToImage):
            raise InvalidModelStructureError("A NeuralNetworkClassifier cannot be used with images as output.")
        elif isinstance(input_conversion, InputConversionImage):
            if not isinstance(output_conversion, _OutputConversionImage):
                raise InvalidModelStructureError(
                    "The defined model uses an input conversion for images but no output conversion for images.",
                )
            data_dimensions = 2
            for layer in layers:
                if data_dimensions == 2 and (isinstance(layer, Convolutional2DLayer | _Pooling2DLayer)):
                    continue
                elif data_dimensions == 2 and isinstance(layer, FlattenLayer):
                    data_dimensions = 1
                elif data_dimensions == 1 and isinstance(layer, ForwardLayer):
                    continue
                else:
                    raise InvalidModelStructureError(
                        (
                            "The 2-dimensional data has to be flattened before using a 1-dimensional layer."
                            if data_dimensions == 2
                            else "You cannot use a 2-dimensional layer with 1-dimensional data."
                        ),
                    )
            if data_dimensions == 2 and (
                isinstance(output_conversion, OutputConversionImageToColumn | OutputConversionImageToTable)
            ):
                raise InvalidModelStructureError(
                    "The output data would be 2-dimensional but the provided output conversion uses 1-dimensional data.",
                )
        elif isinstance(output_conversion, _OutputConversionImage):
            raise InvalidModelStructureError(
                "The defined model uses an output conversion for images but no input conversion for images.",
            )
        else:
            for layer in layers:
                if isinstance(layer, Convolutional2DLayer | FlattenLayer | _Pooling2DLayer):
                    raise InvalidModelStructureError("You cannot use a 2-dimensional layer with 1-dimensional data.")

        self._input_conversion: InputConversion[IFT, IPT] = input_conversion
        self._model = _create_internal_model(input_conversion, layers, is_for_classification=True)
        self._output_conversion: OutputConversion[IPT, OT] = output_conversion
        self._input_size = self._model.input_size
        self._batch_size = 1
        self._is_fitted = False
        self._num_of_classes = (
            layers[-1].output_size if isinstance(layers[-1].output_size, int) else -1
        )  # Is always int but linter doesn't know
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

        _init_default_device()

        if not self._input_conversion._is_fit_data_valid(train_data):
            raise FeatureDataMismatchError
        if epoch_size < 1:
            raise OutOfBoundsError(actual=epoch_size, name="epoch_size", lower_bound=ClosedBound(1))
        if batch_size < 1:
            raise OutOfBoundsError(actual=batch_size, name="batch_size", lower_bound=ClosedBound(1))
        if self._input_conversion._data_size is not self._input_size:
            raise InputSizeError(self._input_conversion._data_size, self._input_size)

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

        _init_default_device()

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
        return self._output_conversion._data_conversion(
            test_data,
            torch.cat(predictions, dim=0),
            **self._input_conversion._get_output_configuration(),
        )

    @property
    def is_fitted(self) -> bool:
        """Whether the model is fitted."""
        return self._is_fitted


def _create_internal_model(
    input_conversion: InputConversion[IFT, IPT],
    layers: list[Layer],
    is_for_classification: bool,
) -> nn.Module:
    from torch import nn

    _init_default_device()

    class _InternalModel(nn.Module):
        def __init__(self, layers: list[Layer], is_for_classification: bool) -> None:

            super().__init__()
            self._layer_list = layers
            internal_layers = []
            previous_output_size = None

            for layer in layers:
                if previous_output_size is not None:
                    layer._set_input_size(previous_output_size)
                elif isinstance(input_conversion, InputConversionImage):
                    layer._set_input_size(input_conversion._data_size)
                if isinstance(layer, FlattenLayer | _Pooling2DLayer):
                    internal_layers.append(layer._get_internal_layer())
                else:
                    internal_layers.append(layer._get_internal_layer(activation_function="relu"))
                previous_output_size = layer.output_size

            if is_for_classification:
                internal_layers.pop()
                if isinstance(layers[-1].output_size, int) and layers[-1].output_size > 2:
                    internal_layers.append(layers[-1]._get_internal_layer(activation_function="none"))
                else:
                    internal_layers.append(layers[-1]._get_internal_layer(activation_function="sigmoid"))
            self._pytorch_layers = nn.Sequential(*internal_layers)

        @property
        def input_size(self) -> int | ImageSize:
            return self._layer_list[0].input_size

        def forward(self, x: Tensor) -> Tensor:
            for layer in self._pytorch_layers:
                x = layer(x)
            return x

    return _InternalModel(layers, is_for_classification)
