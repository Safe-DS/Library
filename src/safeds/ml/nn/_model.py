from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Generic, Literal, Self, TypeVar

from safeds._config import _init_default_device
from safeds._validation import _check_bounds, _ClosedBound
from safeds.data.image.containers import ImageList
from safeds.data.labeled.containers import ImageDataset, TabularDataset, TimeSeriesDataset
from safeds.data.labeled.containers._image_dataset import _ColumnAsTensor
from safeds.data.tabular.containers import Column, Table
from safeds.data.tabular.transformation import OneHotEncoder
from safeds.exceptions import (
    FeatureDataMismatchError,
    FittingWithChoiceError,
    InvalidModelStructureError,
    ModelNotFittedError,
)
from safeds.ml.metrics import ClassificationMetrics, RegressionMetrics
from safeds.ml.nn.converters import (
    InputConversionImageToColumn,
    InputConversionImageToImage,
    InputConversionImageToTable,
)
from safeds.ml.nn.converters._input_converter_image import _InputConversionImage
from safeds.ml.nn.layers import (
    Convolutional2DLayer,
    FlattenLayer,
    ForwardLayer,
)
from safeds.ml.nn.layers._pooling2d_layer import _Pooling2DLayer
from safeds.ml.nn.typing import ConstantImageSize, ModelImageSize, VariableImageSize

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn
    from torch.nn import Module
    from transformers.image_processing_utils import BaseImageProcessor

    from safeds.ml.nn.converters import InputConversion
    from safeds.ml.nn.layers import Layer

IFT = TypeVar("IFT", TabularDataset, TimeSeriesDataset, ImageDataset)  # InputFitType
IPT = TypeVar("IPT", Table, TimeSeriesDataset, ImageList)  # InputPredictType


class NeuralNetworkRegressor(Generic[IFT, IPT]):
    """
    A NeuralNetworkRegressor is a neural network that is used for regression tasks.

    Parameters
    ----------
    input_conversion:
        to convert the input data for the neural network
    layers:
        a list of layers for the neural network to learn

    Raises
    ------
    InvalidModelStructureError
        if the defined model structure is invalid
    """

    def __init__(
        self,
        input_conversion: InputConversion[IFT, IPT],
        layers: list[Layer],
    ):
        if len(layers) == 0:
            raise InvalidModelStructureError("You need to provide at least one layer to a neural network.")
        if isinstance(input_conversion, _InputConversionImage):
            # TODO: why is this limitation needed? we might want to output the probability that an image shows a certain
            #  object, which would be a 1-dimensional output.
            if isinstance(input_conversion, InputConversionImageToColumn | InputConversionImageToTable):
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
            if data_dimensions == 1 and isinstance(input_conversion, InputConversionImageToImage):
                raise InvalidModelStructureError(
                    "The output data would be 1-dimensional but the provided output conversion uses 2-dimensional data.",
                )
        else:
            for layer in layers:
                if isinstance(layer, Convolutional2DLayer | FlattenLayer | _Pooling2DLayer):
                    raise InvalidModelStructureError("You cannot use a 2-dimensional layer with 1-dimensional data.")

        self._input_conversion: InputConversion[IFT, IPT] = input_conversion
        self._model: Module | None = None
        self._layers: list[Layer] = layers
        self._input_size: int | ModelImageSize | None = None
        self._batch_size = 1
        self._is_fitted = False
        self._total_number_of_batches_done = 0
        self._total_number_of_epochs_done = 0

    @staticmethod
    def from_pretrained_model(huggingface_repo: str) -> NeuralNetworkRegressor:  # pragma: no cover
        """
        Load a pretrained model from a [Huggingface repository](https://huggingface.co/models/).

        Parameters
        ----------
        huggingface_repo:
            the name of the huggingface repository

        Returns
        -------
        pretrained_model:
            the pretrained model as a NeuralNetworkRegressor
        """
        from transformers import (
            AutoConfig,
            AutoImageProcessor,
            AutoModelForImageToImage,
            PretrainedConfig,
            Swin2SRForImageSuperResolution,
            Swin2SRImageProcessor,
        )

        _init_default_device()

        config: PretrainedConfig = AutoConfig.from_pretrained(huggingface_repo)

        if config.model_type != "swin2sr":
            raise ValueError("This model is not supported")

        model: Swin2SRForImageSuperResolution = AutoModelForImageToImage.from_pretrained(huggingface_repo)

        image_processor: Swin2SRImageProcessor = AutoImageProcessor.from_pretrained(huggingface_repo)

        if hasattr(config, "num_channels"):
            input_size = VariableImageSize(image_processor.pad_size, image_processor.pad_size, config.num_channels)
        else:  # Should never happen due to model check
            raise ValueError("This model is not supported")  # pragma: no cover

        in_conversion = InputConversionImageToImage(input_size)

        network = NeuralNetworkRegressor.__new__(NeuralNetworkRegressor)
        network._input_conversion = in_conversion
        network._model = model
        network._input_size = input_size
        network._batch_size = 1
        network._is_fitted = True
        network._total_number_of_epochs_done = 0
        network._total_number_of_batches_done = 0

        return network

    def fit(
        self,
        train_data: IFT,
        epoch_count: int = 25,
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
        epoch_count:
            The number of times the training cycle should be done.
        batch_size:
            The size of data batches that should be loaded at one time.
        learning_rate:
            The learning rate of the neural network.
        callback_on_batch_completion:
            Function used to view metrics while training. Gets called after a batch is completed with the index of the
            last batch and the overall loss average.
        callback_on_epoch_completion:
            Function used to view metrics while training. Gets called after an epoch is completed with the index of the
            last epoch and the overall loss average.

        Returns
        -------
        trained_model:
            The trained Model

        Raises
        ------
        OutOfBoundsError
            If epoch_count < 1
            If batch_size < 1
        """
        import torch
        from torch import nn

        from ._internal_model import _InternalModel  # Slow import on global level

        _init_default_device()

        if self._contains_choices():
            raise FittingWithChoiceError

        if not self._input_conversion._is_fit_data_valid(train_data):
            raise FeatureDataMismatchError

        _check_bounds("epoch_count", epoch_count, lower_bound=_ClosedBound(1))
        _check_bounds("batch_size", batch_size, lower_bound=_ClosedBound(1))

        copied_model = copy.deepcopy(self)
        # TODO: How is this supposed to work with pre-trained models? Should the old weights be kept or discarded?
        copied_model._model = _InternalModel(self._input_conversion, self._layers, is_for_classification=False)
        copied_model._input_size = copied_model._model.input_size
        copied_model._batch_size = batch_size

        # TODO: Re-enable or remove depending on how the above TODO is resolved
        # if copied_model._input_conversion._data_size != copied_model._input_size:
        #     raise InputSizeError(copied_model._input_conversion._data_size, copied_model._input_size)

        dataloader = copied_model._input_conversion._data_conversion_fit(train_data, copied_model._batch_size)

        loss_fn = nn.MSELoss()

        optimizer = torch.optim.SGD(copied_model._model.parameters(), lr=learning_rate)
        for _ in range(epoch_count):
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

    # TODO: Does not work if tensorflow with CUDA is used
    # def fit_by_exhaustive_search(
    #     self,
    #     train_data: IFT,
    #     optimization_metric: Literal[
    #         "mean_squared_error",
    #         "mean_absolute_error",
    #         "median_absolute_deviation",
    #         "coefficient_of_determination",
    #     ],
    #     epoch_count: int = 25,
    #     batch_size: int = 1,
    #     learning_rate: float = 0.001,
    # ) -> Self:
    #     """
    #     Use the hyperparameter choices to create multiple models and fit them.
    #
    #     **Note:** This model is not modified.
    #
    #     Parameters
    #     ----------
    #     train_data:
    #         The data the network should be trained on.
    #     optimization_metric:
    #         The metric that should be used for determining the performance of a model.
    #     epoch_count:
    #         The number of times the training cycle should be done.
    #     batch_size:
    #         The size of data batches that should be loaded at one time.
    #     learning_rate:
    #         The learning rate of the neural network.
    #
    #     Returns
    #     -------
    #     best_model:
    #         The model that performed the best out of all possible models given the Choices of hyperparameters.
    #
    #     Raises
    #     ------
    #     FittingWithoutChoiceError
    #         When calling this method on a model without hyperparameter choices.
    #     LearningError
    #         If the training data contains invalid values or if the training failed. Currently raised, when calling this on RNNs or CNNs as well.
    #     """
    #     _init_default_device()
    #
    #     if not self._contains_choices():
    #         raise FittingWithoutChoiceError
    #
    #     if isinstance(train_data, ImageDataset):
    #         raise LearningError(
    #             "Hyperparameter optimization is currently not supported for CNN Regression Tasks.",
    #         )  # pragma: no cover
    #
    #     _check_bounds("epoch_count", epoch_count, lower_bound=_ClosedBound(1))
    #     _check_bounds("batch_size", batch_size, lower_bound=_ClosedBound(1))
    #
    #     list_of_models = self._get_models_for_all_choices()
    #     list_of_fitted_models: list[Self] = []
    #     if isinstance(train_data, TabularDataset):
    #         (train_set, test_set) = self._data_split_table(train_data)
    #     else:
    #         (train_set, test_set) = self._data_split_time_series(train_data)  # type: ignore[assignment]
    #
    #     with ProcessPoolExecutor(max_workers=len(list_of_models), mp_context=mp.get_context("spawn")) as executor:
    #         futures = []
    #         for model in list_of_models:
    #             futures.append(
    #                 executor.submit(
    #                     model.fit,
    #                     train_set,  # type: ignore[arg-type]
    #                     epoch_count,
    #                     batch_size,
    #                     learning_rate,
    #                 ),
    #             )  # type: ignore[arg-type]
    #         [done, _] = wait(futures, return_when=ALL_COMPLETED)
    #         for future in done:
    #             list_of_fitted_models.append(future.result())
    #     executor.shutdown()
    #
    #     if isinstance(train_data, TabularDataset):
    #         return self._get_best_fnn_model(list_of_fitted_models, test_set, optimization_metric)
    #     else:  # train_data is TimeSeriesDataset
    #         return self._get_best_rnn_model(
    #             list_of_fitted_models,
    #             train_set,  # type: ignore[arg-type]
    #             test_set,  # type: ignore[arg-type]
    #             optimization_metric,
    #         )

    def _data_split_table(self, data: TabularDataset) -> tuple[TabularDataset, TabularDataset]:
        [train_split, test_split] = data.to_table().split_rows(0.75)
        train_data = train_split.to_tabular_dataset(
            target_name=data.target.name,
            extra_names=data.extras.column_names,
        )
        test_dataset = test_split.to_tabular_dataset(
            target_name=train_data.target.name,
            extra_names=train_data.extras.column_names,
        )
        return train_data, test_dataset

    def _get_best_fnn_model(
        self,
        list_of_fitted_models: list[Self],
        test_data: TabularDataset,
        optimization_metric: Literal[
            "mean_squared_error",
            "mean_absolute_error",
            "median_absolute_deviation",
            "coefficient_of_determination",
        ],
    ) -> Self:
        test_features = test_data.features
        test_target = test_data.target
        best_model = None
        best_metric_value = None
        for fitted_model in list_of_fitted_models:
            if best_model is None:
                best_model = fitted_model
                match optimization_metric:
                    case "mean_squared_error":
                        best_metric_value = RegressionMetrics.mean_squared_error(
                            predicted=fitted_model.predict(test_features),  # type: ignore[arg-type]
                            expected=test_target,
                        )  # type: ignore[arg-type]
                    case "mean_absolute_error":
                        best_metric_value = RegressionMetrics.mean_absolute_error(
                            predicted=fitted_model.predict(test_features),  # type: ignore[arg-type]
                            expected=test_target,
                        )  # type: ignore[arg-type]
                    case "median_absolute_deviation":
                        best_metric_value = RegressionMetrics.median_absolute_deviation(
                            predicted=fitted_model.predict(test_features),  # type: ignore[arg-type]
                            expected=test_target,
                        )  # type: ignore[arg-type]
                    case "coefficient_of_determination":
                        best_metric_value = RegressionMetrics.coefficient_of_determination(
                            predicted=fitted_model.predict(test_features),  # type: ignore[arg-type]
                            expected=test_target,
                        )  # type: ignore[arg-type]
            else:
                match optimization_metric:
                    case "mean_squared_error":
                        error_of_fitted_model = RegressionMetrics.mean_squared_error(
                            predicted=fitted_model.predict(test_features),  # type: ignore[arg-type]
                            expected=test_target,
                        )  # type: ignore[arg-type]
                        if error_of_fitted_model < best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = error_of_fitted_model  # pragma: no cover
                    case "mean_absolute_error":
                        error_of_fitted_model = RegressionMetrics.mean_absolute_error(
                            predicted=fitted_model.predict(test_features),  # type: ignore[arg-type]
                            expected=test_target,
                        )  # type: ignore[arg-type]
                        if error_of_fitted_model < best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = error_of_fitted_model  # pragma: no cover
                    case "median_absolute_deviation":
                        error_of_fitted_model = RegressionMetrics.median_absolute_deviation(
                            predicted=fitted_model.predict(test_features),  # type: ignore[arg-type]
                            expected=test_target,
                        )  # type: ignore[arg-type]
                        if error_of_fitted_model < best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = error_of_fitted_model  # pragma: no cover
                    case "coefficient_of_determination":
                        error_of_fitted_model = RegressionMetrics.coefficient_of_determination(
                            predicted=fitted_model.predict(test_features),  # type: ignore[arg-type]
                            expected=test_target,
                        )  # type: ignore[arg-type]
                        if error_of_fitted_model > best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = error_of_fitted_model  # pragma: no cover
        assert best_model is not None  # just for linter
        best_model._is_fitted = True
        return best_model

    def _data_split_time_series(self, data: TimeSeriesDataset) -> tuple[TimeSeriesDataset, Table]:
        (train_split, test_split) = data.to_table().split_rows(0.75)
        train_data = train_split.to_time_series_dataset(
            target_name=data.target.name,
            window_size=data.window_size,
            extra_names=data.extras.column_names,
            continuous=data.continuous,
            forecast_horizon=data.forecast_horizon,
        )
        return train_data, test_split

    def _get_best_rnn_model(
        self,
        list_of_fitted_models: list[Self],
        train_data: TimeSeriesDataset,
        test_data: Table,
        optimization_metric: Literal[
            "mean_squared_error",
            "mean_absolute_error",
            "median_absolute_deviation",
            "coefficient_of_determination",
        ],
    ) -> Self:
        test_target = test_data.get_column(train_data.target.name)
        best_model = None
        best_metric_value = None

        size = test_target.row_count
        expected_values = []
        for i in range(size - (train_data.forecast_horizon + train_data.window_size)):
            if train_data.continuous:
                label = test_target[
                    i + train_data.window_size : i + train_data.window_size + train_data.forecast_horizon
                ]
            else:
                label = test_target[i + train_data.window_size + train_data.forecast_horizon]
            expected_values.append(label)
        expected_values_as_col = Column("expected", expected_values)

        for fitted_model in list_of_fitted_models:
            if best_model is None:
                best_model = fitted_model
                match optimization_metric:
                    case "mean_squared_error":
                        best_metric_value = RegressionMetrics.mean_squared_error(
                            predicted=fitted_model.predict(test_data),  # type: ignore[arg-type]
                            expected=expected_values_as_col,
                        )  # type: ignore[arg-type]
                    case "mean_absolute_error":
                        best_metric_value = RegressionMetrics.mean_absolute_error(
                            predicted=fitted_model.predict(test_data),  # type: ignore[arg-type]
                            expected=expected_values_as_col,
                        )  # type: ignore[arg-type]
                    case "median_absolute_deviation":
                        best_metric_value = RegressionMetrics.median_absolute_deviation(
                            predicted=fitted_model.predict(test_data),  # type: ignore[arg-type]
                            expected=expected_values_as_col,
                        )  # type: ignore[arg-type]
                    case "coefficient_of_determination":
                        best_metric_value = RegressionMetrics.coefficient_of_determination(
                            predicted=fitted_model.predict(test_data),  # type: ignore[arg-type]
                            expected=expected_values_as_col,
                        )  # type: ignore[arg-type]
            else:
                match optimization_metric:
                    case "mean_squared_error":
                        error_of_fitted_model = RegressionMetrics.mean_squared_error(
                            predicted=fitted_model.predict(test_data),  # type: ignore[arg-type]
                            expected=expected_values_as_col,
                        )  # type: ignore[arg-type]
                        if error_of_fitted_model < best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = error_of_fitted_model  # pragma: no cover
                    case "mean_absolute_error":
                        error_of_fitted_model = RegressionMetrics.mean_absolute_error(
                            predicted=fitted_model.predict(test_data),  # type: ignore[arg-type]
                            expected=expected_values_as_col,
                        )  # type: ignore[arg-type]
                        if error_of_fitted_model < best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = error_of_fitted_model  # pragma: no cover
                    case "median_absolute_deviation":
                        error_of_fitted_model = RegressionMetrics.median_absolute_deviation(
                            predicted=fitted_model.predict(test_data),  # type: ignore[arg-type]
                            expected=expected_values_as_col,
                        )  # type: ignore[arg-type]
                        if error_of_fitted_model < best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = error_of_fitted_model  # pragma: no cover
                    case "coefficient_of_determination":
                        error_of_fitted_model = RegressionMetrics.coefficient_of_determination(
                            predicted=fitted_model.predict(test_data),  # type: ignore[arg-type]
                            expected=expected_values_as_col,
                        )  # type: ignore[arg-type]
                        if error_of_fitted_model > best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = error_of_fitted_model  # pragma: no cover
        assert best_model is not None  # just for linter
        best_model._is_fitted = True
        return best_model

    # def _data_split_image(self, train_data: ImageDataset) -> (ImageDataset, ImageDataset):
    #    return train_data.split(0.75)

    def _get_models_for_all_choices(self) -> list[Self]:
        all_possible_layer_combinations: list[list] = [[]]
        for layer in self._layers:
            if not layer._contains_choices():
                for item in all_possible_layer_combinations:
                    item.append(layer)
            else:
                updated_combinations = []
                versions_of_one_layer = layer._get_layers_for_all_choices()
                for version in versions_of_one_layer:
                    copy_of_all_current_possible_combinations = copy.deepcopy(all_possible_layer_combinations)
                    for combination in copy_of_all_current_possible_combinations:
                        combination.append(version)
                        updated_combinations.append(combination)
                all_possible_layer_combinations = updated_combinations

        models = []
        for combination in all_possible_layer_combinations:
            new_model = NeuralNetworkRegressor(input_conversion=self._input_conversion, layers=combination)
            models.append(new_model)
        return models  # type: ignore[return-value]

    def predict(self, test_data: IPT) -> IFT:
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

        if not self._is_fitted or self._model is None:
            raise ModelNotFittedError
        if not self._input_conversion._is_predict_data_valid(test_data):
            raise FeatureDataMismatchError
        dataloader = self._input_conversion._data_conversion_predict(test_data, self._batch_size)
        predictions = []
        with torch.no_grad():
            for x in dataloader:
                elem = self._model(x)
                if not isinstance(elem, torch.Tensor) and hasattr(elem, "reconstruction"):
                    elem = elem.reconstruction  # pragma: no cover
                elif not isinstance(elem, torch.Tensor):
                    raise ValueError(f"Output of model has unsupported type: {type(elem)}")  # pragma: no cover
                predictions.append(elem.squeeze(dim=1))
        return self._input_conversion._data_conversion_output(
            test_data,
            torch.cat(predictions, dim=0),
        )

    @property
    def is_fitted(self) -> bool:
        """Whether the model is fitted."""
        return self._is_fitted

    @property
    def input_size(self) -> int | ModelImageSize | None:
        """The input size of the model."""
        # TODO: raise if not fitted, don't return None
        return self._input_size

    def _contains_choices(self) -> bool:
        """Whether the model contains choices in any layer."""
        return any(layer._contains_choices() for layer in self._layers)


class NeuralNetworkClassifier(Generic[IFT, IPT]):
    """
    A NeuralNetworkClassifier is a neural network that is used for classification tasks.

    Parameters
    ----------
    input_conversion:
        to convert the input data for the neural network
    layers:
        a list of layers for the neural network to learn

    Raises
    ------
    InvalidModelStructureError
        if the defined model structure is invalid
    """

    def __init__(
        self,
        input_conversion: InputConversion[IFT, IPT],
        layers: list[Layer],
    ):
        if len(layers) == 0:
            raise InvalidModelStructureError("You need to provide at least one layer to a neural network.")
        if isinstance(input_conversion, InputConversionImageToImage):
            raise InvalidModelStructureError("A NeuralNetworkClassifier cannot be used with images as output.")
        if isinstance(input_conversion, _InputConversionImage) and isinstance(
            input_conversion._input_size,
            VariableImageSize,
        ):
            raise InvalidModelStructureError(
                "A NeuralNetworkClassifier cannot be used with a InputConversionImage that uses a VariableImageSize.",
            )
        elif isinstance(input_conversion, _InputConversionImage):
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
                isinstance(input_conversion, InputConversionImageToColumn | InputConversionImageToTable)
            ):
                raise InvalidModelStructureError(
                    "The output data would be 2-dimensional but the provided output conversion uses 1-dimensional data.",
                )
        else:
            for layer in layers:
                if isinstance(layer, Convolutional2DLayer | FlattenLayer | _Pooling2DLayer):
                    raise InvalidModelStructureError("You cannot use a 2-dimensional layer with 1-dimensional data.")

        self._input_conversion: InputConversion[IFT, IPT] = input_conversion
        self._model: nn.Module | None = None
        self._layers: list[Layer] = layers
        self._input_size: int | ModelImageSize | None = None
        self._batch_size = 1
        self._is_fitted = False
        self._num_of_classes = (
            layers[-1].output_size if isinstance(layers[-1].output_size, int) else -1
        )  # Is always int but linter doesn't know
        self._total_number_of_batches_done = 0
        self._total_number_of_epochs_done = 0

    @staticmethod
    def from_pretrained_model(huggingface_repo: str) -> NeuralNetworkClassifier:  # pragma: no cover
        """
        Load a pretrained model from a [Huggingface repository](https://huggingface.co/models/).

        Parameters
        ----------
        huggingface_repo:
            the name of the huggingface repository

        Returns
        -------
        pretrained_model:
            the pretrained model as a NeuralNetworkClassifier
        """
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification, PretrainedConfig
        from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES

        _init_default_device()

        config: PretrainedConfig = AutoConfig.from_pretrained(huggingface_repo)

        if config.model_type not in MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES:
            raise ValueError("This model is not supported")

        model: Module = AutoModelForImageClassification.from_pretrained(huggingface_repo)

        image_processor: BaseImageProcessor = AutoImageProcessor.from_pretrained(huggingface_repo)
        if hasattr(image_processor, "size") and hasattr(config, "num_channels"):
            if "shortest_edge" in image_processor.size:
                input_size = ConstantImageSize(
                    image_processor.size.get("shortest_edge"),
                    image_processor.size.get("shortest_edge"),
                    config.num_channels,
                )
            else:
                input_size = ConstantImageSize(
                    image_processor.size.get("width"),
                    image_processor.size.get("height"),
                    config.num_channels,
                )
        else:  # Should never happen due to model check
            raise ValueError("This model is not supported")  # pragma: no cover

        label_dict: dict[str, str] = config.id2label
        column_name = "label"
        labels_table = Table({column_name: [label for _, label in label_dict.items()]})
        one_hot_encoder = OneHotEncoder(column_names=[column_name]).fit(labels_table)

        in_conversion = InputConversionImageToColumn(input_size)

        in_conversion._column_name = column_name
        in_conversion._one_hot_encoder = one_hot_encoder
        in_conversion._input_size = input_size
        in_conversion._output_type = _ColumnAsTensor
        num_of_classes = labels_table.row_count

        network = NeuralNetworkClassifier.__new__(NeuralNetworkClassifier)
        network._input_conversion = in_conversion
        network._model = model
        network._input_size = input_size
        network._batch_size = 1
        network._is_fitted = True
        network._num_of_classes = num_of_classes
        network._total_number_of_epochs_done = 0
        network._total_number_of_batches_done = 0

        return network

    def fit(
        self,
        train_data: IFT,
        epoch_count: int = 25,
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
        epoch_count:
            The number of times the training cycle should be done.
        batch_size:
            The size of data batches that should be loaded at one time.
        learning_rate:
            The learning rate of the neural network.
        callback_on_batch_completion:
            Function used to view metrics while training. Gets called after a batch is completed with the index of the
            last batch and the overall loss average.
        callback_on_epoch_completion:
            Function used to view metrics while training. Gets called after an epoch is completed with the index of the
            last epoch and the overall loss average.

        Returns
        -------
        trained_model:
            The trained Model

        Raises
        ------
        ValueError
            If epoch_count < 1
            If batch_size < 1
        """
        import torch
        from torch import nn

        from ._internal_model import _InternalModel  # Slow import on global level

        _init_default_device()

        if self._contains_choices():
            raise FittingWithChoiceError

        if isinstance(train_data, TimeSeriesDataset) and train_data.continuous:
            raise NotImplementedError(
                "Continuous Predictions are currently not supported for Time Series Classification.",
            )

        if not self._input_conversion._is_fit_data_valid(train_data):
            raise FeatureDataMismatchError

        _check_bounds("epoch_count", epoch_count, lower_bound=_ClosedBound(1))
        _check_bounds("batch_size", batch_size, lower_bound=_ClosedBound(1))

        copied_model = copy.deepcopy(self)
        # TODO: How is this supposed to work with pre-trained models? Should the old weights be kept or discarded?
        copied_model._model = _InternalModel(self._input_conversion, self._layers, is_for_classification=True)
        copied_model._batch_size = batch_size
        copied_model._input_size = copied_model._model.input_size

        # TODO: Re-enable or remove depending on how the above TODO is resolved
        # if copied_model._input_conversion._data_size != copied_model._input_size:
        #     raise InputSizeError(copied_model._input_conversion._data_size, copied_model._input_size)

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
        for _ in range(epoch_count):
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

    # TODO: Does not work if tensorflow with CUDA is used
    # def fit_by_exhaustive_search(
    #     self,
    #     train_data: IFT,
    #     optimization_metric: Literal["accuracy", "precision", "recall", "f1_score"],
    #     positive_class: Any = None,
    #     epoch_count: int = 25,
    #     batch_size: int = 1,
    #     learning_rate: float = 0.001,
    # ) -> Self:
    #     """
    #     Use the hyperparameter choices to create multiple models and fit them.
    #
    #     **Note:** This model is not modified.
    #
    #     Parameters
    #     ----------
    #     train_data:
    #         The data the network should be trained on.
    #     optimization_metric:
    #         The metric that should be used for determining the performance of a model.
    #     positive_class:
    #         The class to be considered positive. Only needs to be provided when choosing precision, recall or f1_score as the optimization metric.
    #     epoch_count:
    #         The number of times the training cycle should be done.
    #     batch_size:
    #         The size of data batches that should be loaded at one time.
    #     learning_rate:
    #         The learning rate of the neural network.
    #
    #     Returns
    #     -------
    #     best_model:
    #         The model that performed the best out of all possible models given the Choices of hyperparameters.
    #
    #     Raises
    #     ------
    #     FittingWithoutChoiceError
    #         When calling this method on a model without hyperparameter choices.
    #     LearningError
    #         If the training data contains invalid values or if the training failed.
    #     """
    #     _init_default_device()
    #
    #     if not self._contains_choices():
    #         raise FittingWithoutChoiceError
    #
    #     if isinstance(train_data, TimeSeriesDataset) and train_data.continuous:
    #         raise NotImplementedError(
    #             "Continuous Predictions are currently not supported for Time Series Classification.",
    #         )
    #
    #     _check_bounds("epoch_count", epoch_count, lower_bound=_ClosedBound(1))
    #     _check_bounds("batch_size", batch_size, lower_bound=_ClosedBound(1))
    #
    #     list_of_models = self._get_models_for_all_choices()
    #     list_of_fitted_models: list[Self] = []
    #
    #     if isinstance(train_data, TabularDataset):
    #         (train_set, test_set) = self._data_split_table(train_data)
    #     elif isinstance(train_data, TimeSeriesDataset):
    #         (train_set, test_set) = self._data_split_time_series(train_data)  # type: ignore[assignment]
    #     else:  # train_data is ImageDataset
    #         (train_set, test_set) = self._data_split_image(train_data)  # type: ignore[assignment]
    #
    #     with ProcessPoolExecutor(max_workers=len(list_of_models), mp_context=mp.get_context("spawn")) as executor:
    #         futures = []
    #         for model in list_of_models:
    #             futures.append(
    #                 executor.submit(
    #                     model.fit,
    #                     train_set,  # type: ignore[arg-type]
    #                     epoch_count,
    #                     batch_size,
    #                     learning_rate,
    #                 ),
    #             )  # type: ignore[arg-type]
    #         [done, _] = wait(futures, return_when=ALL_COMPLETED)
    #         for future in done:
    #             list_of_fitted_models.append(future.result())
    #     executor.shutdown()
    #
    #     if isinstance(train_data, TabularDataset):
    #         return self._get_best_fnn_model(list_of_fitted_models, test_set, optimization_metric, positive_class)
    #     elif isinstance(train_data, TimeSeriesDataset):
    #         return self._get_best_rnn_model(
    #             list_of_fitted_models,
    #             train_set,  # type: ignore[arg-type]
    #             test_set,  # type: ignore[arg-type]
    #             optimization_metric,
    #             positive_class,
    #         )
    #     elif isinstance(self._input_conversion, InputConversionImageToColumn):
    #         return self._get_best_cnn_model_column(
    #             list_of_fitted_models,
    #             train_set,  # type: ignore[arg-type]
    #             optimization_metric,
    #             positive_class,
    #         )
    #     else:  # ImageToTable
    #         return self._get_best_cnn_model_table(
    #             list_of_fitted_models,
    #             train_set,  # type: ignore[arg-type]
    #             optimization_metric,
    #             positive_class,
    #         )

    def _data_split_table(self, data: TabularDataset) -> tuple[TabularDataset, TabularDataset]:
        [train_split, test_split] = data.to_table().split_rows(0.75)
        train_data = train_split.to_tabular_dataset(
            target_name=data.target.name,
            extra_names=data.extras.column_names,
        )
        test_data = test_split.to_tabular_dataset(
            target_name=train_data.target.name,
            extra_names=train_data.extras.column_names,
        )
        return (train_data, test_data)

    def _data_split_time_series(self, data: TimeSeriesDataset) -> tuple[TimeSeriesDataset, Table]:
        (train_split, test_split) = data.to_table().split_rows(0.75)
        train_data = train_split.to_time_series_dataset(
            target_name=data.target.name,
            window_size=data.window_size,
            extra_names=data.extras.column_names,
            continuous=data.continuous,
            forecast_horizon=data.forecast_horizon,
        )
        return train_data, test_split

    def _data_split_image(self, train_data: ImageDataset) -> tuple[ImageDataset, ImageDataset]:
        return train_data.split(0.75)

    def _get_best_fnn_model(
        self,
        list_of_fitted_models: list[Self],
        test_data: TabularDataset,
        optimization_metric: Literal["accuracy", "precision", "recall", "f1_score"],
        positive_class: Any = None,
    ) -> Self:
        test_features = test_data.features
        test_target = test_data.target
        best_model = None
        best_metric_value = None
        for fitted_model in list_of_fitted_models:
            if best_model is None:
                best_model = fitted_model
                match optimization_metric:
                    case "accuracy":
                        best_metric_value = ClassificationMetrics.accuracy(
                            predicted=fitted_model.predict(test_features),  # type: ignore[arg-type]
                            expected=test_target,
                        )  # type: ignore[arg-type]
                    case "precision":
                        best_metric_value = ClassificationMetrics.precision(
                            predicted=fitted_model.predict(test_features),  # type: ignore[arg-type]
                            expected=test_target,  # type: ignore[arg-type]
                            positive_class=positive_class,
                        )  # type: ignore[arg-type]
                    case "recall":
                        best_metric_value = ClassificationMetrics.recall(
                            predicted=fitted_model.predict(test_features),  # type: ignore[arg-type]
                            expected=test_target,  # type: ignore[arg-type]
                            positive_class=positive_class,
                        )  # type: ignore[arg-type]
                    case "f1_score":
                        best_metric_value = ClassificationMetrics.f1_score(
                            predicted=fitted_model.predict(test_features),  # type: ignore[arg-type]
                            expected=test_target,  # type: ignore[arg-type]
                            positive_class=positive_class,
                        )  # type: ignore[arg-type]
            else:
                match optimization_metric:
                    case "accuracy":
                        error_of_fitted_model = ClassificationMetrics.accuracy(
                            predicted=fitted_model.predict(test_features),  # type: ignore[arg-type]
                            expected=test_target,
                        )
                        if error_of_fitted_model > best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = error_of_fitted_model  # pragma: no cover
                    case "precision":
                        error_of_fitted_model = ClassificationMetrics.precision(
                            predicted=fitted_model.predict(test_features),
                            expected=test_target,
                            # type: ignore[arg-type]
                            positive_class=positive_class,
                        )  # type: ignore[arg-type]
                        if error_of_fitted_model > best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = error_of_fitted_model  # pragma: no cover
                    case "recall":
                        error_of_fitted_model = ClassificationMetrics.recall(
                            predicted=fitted_model.predict(test_features),  # type: ignore[arg-type]
                            expected=test_target,
                            positive_class=positive_class,
                        )  # type: ignore[arg-type]
                        if error_of_fitted_model > best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = error_of_fitted_model  # pragma: no cover
                    case "f1_score":
                        error_of_fitted_model = ClassificationMetrics.f1_score(
                            predicted=fitted_model.predict(test_features),
                            expected=test_target,
                            # type: ignore[arg-type]
                            positive_class=positive_class,
                        )  # type: ignore[arg-type]
                        if error_of_fitted_model > best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = error_of_fitted_model  # pragma: no cover
        assert best_model is not None  # just for linter
        best_model._is_fitted = True
        return best_model

    def _get_best_rnn_model(
        self,
        list_of_fitted_models: list[Self],
        train_data: TimeSeriesDataset,
        test_data: Table,
        optimization_metric: Literal["accuracy", "precision", "recall", "f1_score"],
        positive_class: Any = None,
    ) -> Self:
        test_target = test_data.get_column(train_data.target.name)

        size = test_target.row_count
        expected_values = []
        for i in range(size - (train_data.forecast_horizon + train_data.window_size)):
            label = test_target[i + train_data.window_size + train_data.forecast_horizon]
            expected_values.append(label)
        expected_values_as_col = Column("expected", expected_values)

        best_model = None
        best_metric_value = None
        for fitted_model in list_of_fitted_models:
            if best_model is None:
                best_model = fitted_model
                match optimization_metric:
                    case "accuracy":
                        best_metric_value = ClassificationMetrics.accuracy(
                            predicted=fitted_model.predict(test_data),  # type: ignore[arg-type]
                            expected=expected_values_as_col,
                        )  # type: ignore[arg-type]
                    case "precision":
                        best_metric_value = ClassificationMetrics.precision(
                            predicted=fitted_model.predict(test_data),  # type: ignore[arg-type]
                            expected=expected_values_as_col,
                            positive_class=positive_class,
                        )  # type: ignore[arg-type]
                    case "recall":
                        best_metric_value = ClassificationMetrics.recall(
                            predicted=fitted_model.predict(test_data),  # type: ignore[arg-type]
                            expected=expected_values_as_col,
                            positive_class=positive_class,
                        )  # type: ignore[arg-type]
                    case "f1_score":
                        best_metric_value = ClassificationMetrics.f1_score(
                            predicted=fitted_model.predict(test_data),  # type: ignore[arg-type]
                            expected=expected_values_as_col,
                            positive_class=positive_class,
                        )  # type: ignore[arg-type]
            else:
                match optimization_metric:
                    case "accuracy":
                        error_of_fitted_model = ClassificationMetrics.accuracy(
                            predicted=fitted_model.predict(test_data),  # type: ignore[arg-type]
                            expected=expected_values_as_col,
                        )  # type: ignore[arg-type]
                        if error_of_fitted_model > best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = error_of_fitted_model  # pragma: no cover
                    case "precision":
                        error_of_fitted_model = ClassificationMetrics.precision(
                            predicted=fitted_model.predict(test_data),
                            expected=expected_values_as_col,
                            # type: ignore[arg-type]
                            positive_class=positive_class,
                        )  # type: ignore[arg-type]
                        if error_of_fitted_model > best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = error_of_fitted_model  # pragma: no cover
                    case "recall":
                        error_of_fitted_model = ClassificationMetrics.recall(
                            predicted=fitted_model.predict(test_data),  # type: ignore[arg-type]
                            expected=expected_values_as_col,
                            positive_class=positive_class,
                        )  # type: ignore[arg-type]
                        if error_of_fitted_model > best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = error_of_fitted_model  # pragma: no cover
                    case "f1_score":
                        error_of_fitted_model = ClassificationMetrics.f1_score(
                            predicted=fitted_model.predict(test_data),
                            expected=expected_values_as_col,
                            # type: ignore[arg-type]
                            positive_class=positive_class,
                        )  # type: ignore[arg-type]
                        if error_of_fitted_model > best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = error_of_fitted_model  # pragma: no cover
        assert best_model is not None  # just for linter
        best_model._is_fitted = True
        return best_model

    def _get_best_cnn_model_column(
        self,
        list_of_fitted_models: list[Self],
        test_data: ImageDataset,
        optimization_metric: Literal["accuracy", "precision", "recall", "f1_score"],
        positive_class: Any = None,
    ) -> Self:
        input_data = test_data.get_input()
        expected = test_data.get_output()
        best_model = None
        best_metric_value = None
        for fitted_model in list_of_fitted_models:
            prediction = fitted_model.predict(input_data).get_output()  # type: ignore[attr-defined, arg-type]
            if best_model is None:
                best_model = fitted_model
                match optimization_metric:
                    case "accuracy":
                        best_metric_value = ClassificationMetrics.accuracy(predicted=prediction, expected=expected)
                    case "precision":
                        best_metric_value = ClassificationMetrics.precision(
                            predicted=prediction,
                            expected=expected,
                            positive_class=positive_class,
                        )
                    case "recall":
                        best_metric_value = ClassificationMetrics.recall(
                            predicted=prediction,
                            expected=expected,
                            positive_class=positive_class,
                        )
                    case "f1_score":
                        best_metric_value = ClassificationMetrics.f1_score(
                            predicted=prediction,
                            expected=expected,
                            positive_class=positive_class,
                        )
            else:
                match optimization_metric:
                    case "accuracy":
                        error_of_fitted_model = ClassificationMetrics.accuracy(predicted=prediction, expected=expected)
                        if error_of_fitted_model > best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = error_of_fitted_model  # pragma: no cover
                    case "precision":
                        error_of_fitted_model = ClassificationMetrics.precision(
                            predicted=prediction,
                            expected=expected,
                            positive_class=positive_class,
                        )
                        if error_of_fitted_model > best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = error_of_fitted_model  # pragma: no cover
                    case "recall":
                        error_of_fitted_model = ClassificationMetrics.recall(
                            predicted=prediction,
                            expected=expected,
                            positive_class=positive_class,
                        )
                        if error_of_fitted_model > best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = error_of_fitted_model  # pragma: no cover
                    case "f1_score":
                        error_of_fitted_model = ClassificationMetrics.f1_score(
                            predicted=prediction,
                            expected=expected,
                            positive_class=positive_class,
                        )
                        if error_of_fitted_model > best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = error_of_fitted_model  # pragma: no cover
        assert best_model is not None  # just for linter
        best_model._is_fitted = True
        return best_model

    def _get_best_cnn_model_table(
        self,
        list_of_fitted_models: list[Self],
        test_data: ImageDataset,
        optimization_metric: Literal["accuracy", "precision", "recall", "f1_score"],
        positive_class: Any = None,
    ) -> Self:
        input_data = test_data.get_input()
        labels = test_data.get_output()
        expected = self._inverse_one_hot_encode_by_index_of_column(labels)

        # Set positive class to index of positive column, to be able to calculate Classification Metrics
        if positive_class is not None:
            for column_index in range(labels.column_count):
                # TODO Find out if a column is the positive class, maybe by the column name?
                if labels.column_names[column_index] == positive_class:
                    positive_class = labels.column_names[column_index]
                    break

        best_model = None
        best_metric_value = None
        for fitted_model in list_of_fitted_models:
            prediction = self._inverse_one_hot_encode_by_index_of_column(
                fitted_model.predict(input_data).get_output(),  # type: ignore[attr-defined, arg-type]
            )
            if best_model is None:
                best_model = fitted_model
                match optimization_metric:
                    case "accuracy":
                        best_metric_value = ClassificationMetrics.accuracy(
                            predicted=prediction,
                            expected=expected,  # type: ignore[arg-type]
                        )  # type: ignore[arg-type]
                    case "precision":
                        best_metric_value = ClassificationMetrics.precision(
                            predicted=prediction,  # type: ignore[arg-type]
                            expected=expected,
                            positive_class=positive_class,
                        )  # type: ignore[arg-type]
                    case "recall":
                        best_metric_value = ClassificationMetrics.recall(
                            predicted=prediction,  # type: ignore[arg-type]
                            expected=expected,
                            positive_class=positive_class,
                        )  # type: ignore[arg-type]
                    case "f1_score":
                        best_metric_value = ClassificationMetrics.f1_score(
                            predicted=prediction,  # type: ignore[arg-type]
                            expected=expected,
                            positive_class=positive_class,
                        )  # type: ignore[arg-type]
            else:
                match optimization_metric:
                    case "accuracy":
                        error_of_fitted_model = ClassificationMetrics.accuracy(
                            predicted=prediction,
                            expected=expected,  # type: ignore[arg-type]
                        )  # type: ignore[arg-type]
                        if error_of_fitted_model > best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = error_of_fitted_model  # pragma: no cover
                    case "precision":
                        error_of_fitted_model = ClassificationMetrics.precision(
                            predicted=prediction,
                            expected=expected,  # type: ignore[arg-type]
                            positive_class=positive_class,
                        )  # type: ignore[arg-type]
                        if error_of_fitted_model > best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = error_of_fitted_model  # pragma: no cover
                    case "recall":
                        error_of_fitted_model = ClassificationMetrics.recall(
                            predicted=prediction,  # type: ignore[arg-type]
                            expected=expected,
                            positive_class=positive_class,
                        )  # type: ignore[arg-type]
                        if error_of_fitted_model > best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = error_of_fitted_model  # pragma: no cover
                    case "f1_score":
                        error_of_fitted_model = ClassificationMetrics.f1_score(
                            predicted=prediction,
                            expected=expected,  # type: ignore[arg-type]
                            positive_class=positive_class,
                        )  # type: ignore[arg-type]
                        if error_of_fitted_model > best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = error_of_fitted_model  # pragma: no cover
        assert best_model is not None  # just for linter
        best_model._is_fitted = True
        return best_model

    def _inverse_one_hot_encode_by_index_of_column(self, table: Table) -> Column:
        indices = []
        for row_index in range(table.row_count):
            for column_index in range(table.column_count):
                if table.get_column(table.column_names[column_index]).get_value(row_index) == 1.0:
                    indices.append(column_index)
                    break
        return Column("class_index", indices)

    def _get_models_for_all_choices(self) -> list[Self]:
        all_possible_layer_combinations: list[list] = [[]]
        for layer in self._layers:
            if not layer._contains_choices():
                for item in all_possible_layer_combinations:
                    item.append(layer)
            else:
                updated_combinations = []
                versions_of_one_layer = layer._get_layers_for_all_choices()
                for version in versions_of_one_layer:
                    copy_of_all_current_possible_combinations = copy.deepcopy(all_possible_layer_combinations)
                    for combination in copy_of_all_current_possible_combinations:
                        combination.append(version)
                        updated_combinations.append(combination)
                all_possible_layer_combinations = updated_combinations

        models = []
        for combination in all_possible_layer_combinations:
            new_model = NeuralNetworkClassifier(input_conversion=self._input_conversion, layers=combination)
            models.append(new_model)
        return models  # type: ignore[return-value]

    def predict(self, test_data: IPT) -> IFT:
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

        if not self._is_fitted or self._model is None:
            raise ModelNotFittedError
        if not self._input_conversion._is_predict_data_valid(test_data):
            raise FeatureDataMismatchError
        dataloader = self._input_conversion._data_conversion_predict(test_data, self._batch_size)
        predictions = []
        with torch.no_grad():
            for x in dataloader:
                elem = self._model(x)
                if not isinstance(elem, torch.Tensor) and hasattr(elem, "logits"):
                    elem = elem.logits  # pragma: no cover
                elif not isinstance(elem, torch.Tensor):
                    raise ValueError(f"Output of model has unsupported type: {type(elem)}")  # pragma: no cover
                if self._num_of_classes > 1:
                    predictions.append(torch.argmax(elem, dim=1))
                else:
                    predictions.append(elem.squeeze(dim=1).round())
        return self._input_conversion._data_conversion_output(
            test_data,
            torch.cat(predictions, dim=0),
        )

    @property
    def is_fitted(self) -> bool:
        """Whether the model is fitted."""
        return self._is_fitted

    @property
    def input_size(self) -> int | ModelImageSize | None:
        """The input size of the model."""
        # TODO: raise if not fitted, don't return None
        return self._input_size

    def _contains_choices(self) -> bool:
        """Whether the model contains choices in any layer."""
        return any(layer._contains_choices() for layer in self._layers)
