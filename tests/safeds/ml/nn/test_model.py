import pickle
import re

import pytest
from torch.types import Device

from safeds.data.image.typing import ImageSize
from safeds.data.labeled.containers import TabularDataset, TimeSeriesDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import (
    FeatureDataMismatchError,
    FittingWithChoiceError,
    InvalidFitDataError,
    InvalidModelStructureError,
    NotFittedError,
    OutOfBoundsError,
)
from safeds.ml.hyperparameters import Choice
from safeds.ml.nn import (
    NeuralNetworkClassifier,
    NeuralNetworkRegressor,
)
from safeds.ml.nn.converters import (
    InputConversion,
    InputConversionImageToColumn,
    InputConversionImageToImage,
    InputConversionImageToTable,
    InputConversionTable,
    InputConversionTimeSeries,
)
from safeds.ml.nn.layers import (
    AveragePooling2DLayer,
    Convolutional2DLayer,
    ConvolutionalTranspose2DLayer,
    FlattenLayer,
    ForwardLayer,
    Layer,
    LSTMLayer,
    MaxPooling2DLayer,
)
from safeds.ml.nn.typing import VariableImageSize
from tests.helpers import configure_test_with_device, get_devices, get_devices_ids


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestClassificationModel:
    class TestFit:
        def test_should_return_input_size(self, device: Device) -> None:
            configure_test_with_device(device)
            model = NeuralNetworkClassifier(
                InputConversionTable(),
                [ForwardLayer(neuron_count=1)],
            ).fit(
                Table.from_dict({"a": [1], "b": [2]}).to_tabular_dataset("a"),
            )
            assert model.input_size == 1

        def test_should_raise_if_epoch_count_out_of_bounds(self, device: Device) -> None:
            invalid_epoch_count = 0
            configure_test_with_device(device)
            with pytest.raises(OutOfBoundsError):
                NeuralNetworkClassifier(
                    InputConversionTable(),
                    [ForwardLayer(1)],
                ).fit(
                    Table.from_dict({"a": [1], "b": [2]}).to_tabular_dataset("a"),
                    epoch_count=invalid_epoch_count,
                )

        def test_should_raise_if_batch_size_out_of_bounds(self, device: Device) -> None:
            invalid_batch_size = 0
            configure_test_with_device(device)
            with pytest.raises(OutOfBoundsError):
                NeuralNetworkClassifier(
                    InputConversionTable(),
                    [ForwardLayer(neuron_count=1)],
                ).fit(
                    Table.from_dict({"a": [1], "b": [2]}).to_tabular_dataset("a"),
                    batch_size=invalid_batch_size,
                )

        def test_should_raise_if_fit_function_returns_wrong_datatype(self, device: Device) -> None:
            configure_test_with_device(device)
            fitted_model = NeuralNetworkClassifier(
                InputConversionTable(),
                [ForwardLayer(neuron_count=2), ForwardLayer(neuron_count=1)],
            ).fit(
                Table.from_dict({"a": [1], "b": [0]}).to_tabular_dataset("a"),
            )
            assert isinstance(fitted_model, NeuralNetworkClassifier)

        def test_should_raise_when_fitting_with_choice(self, device: Device) -> None:
            configure_test_with_device(device)
            model = NeuralNetworkClassifier(InputConversionTable(), [ForwardLayer(Choice(1, 2))])
            with pytest.raises(FittingWithChoiceError):
                model.fit(Table.from_dict({"a": [1], "b": [0]}).to_tabular_dataset("a"))

        def test_should_raise_if_is_fitted_is_set_correctly_for_binary_classification(self, device: Device) -> None:
            configure_test_with_device(device)
            model = NeuralNetworkClassifier(
                InputConversionTable(),
                [ForwardLayer(neuron_count=1)],
            )
            model_2 = NeuralNetworkClassifier(
                InputConversionTable(),
                [LSTMLayer(neuron_count=1)],
            )
            assert not model.is_fitted
            assert not model_2.is_fitted
            model = model.fit(
                Table.from_dict({"a": [1], "b": [0]}).to_tabular_dataset("a"),
            )
            model_2 = model_2.fit(
                Table.from_dict({"a": [1], "b": [0]}).to_tabular_dataset("a"),
            )
            assert model.is_fitted
            assert model_2.is_fitted

        def test_should_raise_if_is_fitted_is_set_correctly_for_multiclass_classification(self, device: Device) -> None:
            configure_test_with_device(device)
            model = NeuralNetworkClassifier(
                InputConversionTable(),
                [ForwardLayer(neuron_count=1), ForwardLayer(neuron_count=3)],
            )
            model_2 = NeuralNetworkClassifier(
                InputConversionTable(),
                [ForwardLayer(neuron_count=1), LSTMLayer(neuron_count=3)],
            )
            assert not model.is_fitted
            assert not model_2.is_fitted
            model = model.fit(
                Table.from_dict({"a": [1, 0, 2], "b": [0, 15, 5]}).to_tabular_dataset("a"),
            )
            model_2 = model_2.fit(
                Table.from_dict({"a": [1, 0, 2], "b": [0, 15, 5]}).to_tabular_dataset("a"),
            )
            assert model.is_fitted
            assert model_2.is_fitted

        def test_should_raise_if_train_features_mismatch(self, device: Device) -> None:
            configure_test_with_device(device)
            model = NeuralNetworkClassifier(
                InputConversionTable(),
                [ForwardLayer(neuron_count=1), ForwardLayer(neuron_count=1)],
            )
            learned_model = model.fit(
                Table.from_dict({"a": [0.1, 0, 0.2], "b": [0, 0.15, 0.5]}).to_tabular_dataset("b"),
            )
            with pytest.raises(
                FeatureDataMismatchError,
                match="The features in the given table do not match with the specified feature columns names of the model.",
            ):
                learned_model.fit(Table.from_dict({"k": [0.1, 0, 0.2], "l": [0, 0.15, 0.5]}).to_tabular_dataset("k"))

        @pytest.mark.parametrize(
            ("table", "reason"),
            [
                (
                    Table.from_dict({"a": [1, 2, 3], "b": [1, 2, None], "c": [0, 15, 5]}).to_tabular_dataset("c"),
                    re.escape("The given Fit Data is invalid:\nThe following Columns contain missing values: ['b']\n"),
                ),
                (
                    Table.from_dict({"a": ["a", "b", "c"], "b": [1, 2, 3], "c": [0, 15, 5]}).to_tabular_dataset("c"),
                    re.escape(
                        "The given Fit Data is invalid:\nThe following Columns contain non-numerical data: ['a']",
                    ),
                ),
                (
                    Table.from_dict({"a": ["a", "b", "c"], "b": [1, 2, None], "c": [0, 15, 5]}).to_tabular_dataset("c"),
                    re.escape(
                        "The given Fit Data is invalid:\nThe following Columns contain missing values: ['b']\nThe following Columns contain non-numerical data: ['a']",
                    ),
                ),
                (
                    Table.from_dict({"a": [1, 2, 3], "b": [1, 2, 3], "c": [0, None, 5]}).to_tabular_dataset("c"),
                    re.escape(
                        "The given Fit Data is invalid:\nThe following Columns contain missing values: ['c']\n",
                    ),
                ),
                (
                    Table.from_dict({"a": [1, 2, 3], "b": [1, 2, 3], "c": ["a", "b", "a"]}).to_tabular_dataset("c"),
                    re.escape(
                        "The given Fit Data is invalid:\nThe following Columns contain non-numerical data: ['c']",
                    ),
                ),
            ],
            ids=[
                "missing value feature",
                "non-numerical feature",
                "missing value and non-numerical features",
                "missing value target",
                "non-numerical target",
            ],
        )
        def test_should_catch_invalid_fit_data(self, device: Device, table: TabularDataset, reason: str) -> None:
            configure_test_with_device(device)
            model = NeuralNetworkClassifier(
                InputConversionTable(),
                [ForwardLayer(neuron_count=4), ForwardLayer(1)],
            )
            with pytest.raises(
                InvalidFitDataError,
                match=reason,
            ):
                model.fit(table)

        def test_should_raise_when_time_series_classification_with_continuous_data(self, device: Device) -> None:
            configure_test_with_device(device)
            # data = Table.from_dict({"a": [1, 2, 3], "b": [1, 2, 3], "c": [0, 1, 0]}).to_time_series_dataset(
            #     "c",
            #     1,
            #     continuous=True,
            # )
            data = TimeSeriesDataset(
                Table.from_dict({"a": [1, 2, 3], "b": [1, 2, 3], "c": [0, 1, 0]}),
                "c",
                1,
                continuous=True,
            )
            model = NeuralNetworkClassifier(
                InputConversionTimeSeries(),
                [ForwardLayer(neuron_count=4), LSTMLayer(neuron_count=3)],
            )
            with pytest.raises(
                NotImplementedError,
                match="Continuous Predictions are currently not supported for Time Series Classification.",
            ):
                model.fit(data)

        def test_should_raise_if_fit_doesnt_batch_callback(self, device: Device) -> None:
            configure_test_with_device(device)
            model = NeuralNetworkClassifier(
                InputConversionTable(),
                [ForwardLayer(neuron_count=1)],
            )

            class Test:
                self.was_called = False

                def cb(self, ind: int, loss: float) -> None:
                    if ind >= 0 and loss >= 0.0:
                        self.was_called = True

                def callback_was_called(self) -> bool:
                    return self.was_called

            obj = Test()
            model.fit(
                Table.from_dict({"a": [1], "b": [0]}).to_tabular_dataset("a"),
                callback_on_batch_completion=obj.cb,
            )

            assert obj.callback_was_called() is True

        def test_should_raise_if_fit_doesnt_epoch_callback(self, device: Device) -> None:
            configure_test_with_device(device)
            model = NeuralNetworkClassifier(
                InputConversionTable(),
                [ForwardLayer(neuron_count=1)],
            )

            class Test:
                self.was_called = False

                def cb(self, ind: int, loss: float) -> None:
                    if ind >= 0 and loss >= 0.0:
                        self.was_called = True

                def callback_was_called(self) -> bool:
                    return self.was_called

            obj = Test()
            model.fit(
                Table.from_dict({"a": [1], "b": [0]}).to_tabular_dataset("a"),
                callback_on_epoch_completion=obj.cb,
            )

            assert obj.callback_was_called() is True

    # TODO: Does not work if tensorflow with CUDA is used
    # class TestFitByExhaustiveSearch:
    #     def test_should_return_input_size(self, device: Device) -> None:
    #         configure_test_with_device(device)
    #
    #         model = NeuralNetworkClassifier(
    #             InputConversionTable(),
    #             [ForwardLayer(neuron_count=Choice(2, 4)), ForwardLayer(1)],
    #         ).fit_by_exhaustive_search(
    #             Table.from_dict({"a": [1, 2, 3, 4], "b": [0, 1, 0, 1]}).to_tabular_dataset("b"),
    #             "accuracy",
    #         )
    #         assert model.input_size == 1
    #
    #     def test_should_raise_if_epoch_count_out_of_bounds_when_fitting_by_exhaustive_search(
    #         self,
    #         device: Device,
    #     ) -> None:
    #         invalid_epoch_count = 0
    #         configure_test_with_device(device)
    #         with pytest.raises(OutOfBoundsError):
    #             NeuralNetworkClassifier(
    #                 InputConversionTable(),
    #                 [ForwardLayer(Choice(2, 4)), ForwardLayer(1)],
    #             ).fit_by_exhaustive_search(
    #                 Table.from_dict({"a": [1], "b": [0]}).to_tabular_dataset("b"),
    #                 "accuracy",
    #                 epoch_count=invalid_epoch_count,
    #             )
    #
    #     def test_should_raise_if_batch_size_out_of_bounds_when_fitting_by_exhaustive_search(
    #         self,
    #         device: Device,
    #     ) -> None:
    #         invalid_batch_size = 0
    #         configure_test_with_device(device)
    #         with pytest.raises(OutOfBoundsError):
    #             NeuralNetworkClassifier(
    #                 InputConversionTable(),
    #                 [ForwardLayer(neuron_count=Choice(2, 4)), ForwardLayer(1)],
    #             ).fit_by_exhaustive_search(
    #                 Table.from_dict({"a": [1], "b": [0]}).to_tabular_dataset("b"),
    #                 "accuracy",
    #                 batch_size=invalid_batch_size,
    #             )
    #
    #     def test_should_raise_when_fitting_by_exhaustive_search_without_choice(self, device: Device) -> None:
    #         configure_test_with_device(device)
    #         model = NeuralNetworkClassifier(InputConversionTable(), [ForwardLayer(1)])
    #         with pytest.raises(FittingWithoutChoiceError):
    #             model.fit_by_exhaustive_search(
    #                 Table.from_dict({"a": [1], "b": [0]}).to_tabular_dataset("b"),
    #                 "accuracy",
    #             )
    #
    #     def test_should_raise_when_time_series_classification_with_continuous_data(self, device: Device) -> None:
    #         configure_test_with_device(device)
    #         data = Table.from_dict({"a": [1, 2, 3], "b": [1, 2, 3], "c": [0, 1, 0]}).to_time_series_dataset(
    #             "c",
    #             1,
    #             continuous=True,
    #         )
    #         model = NeuralNetworkClassifier(
    #             InputConversionTimeSeries(),
    #             [ForwardLayer(neuron_count=Choice(2, 4)), LSTMLayer(neuron_count=3)],
    #         )
    #         with pytest.raises(
    #             NotImplementedError,
    #             match="Continuous Predictions are currently not supported for Time Series Classification.",
    #         ):
    #             model.fit_by_exhaustive_search(data, "accuracy")
    #
    #     @pytest.mark.parametrize(
    #         ("metric", "positive_class"),
    #         [
    #             (
    #                 "accuracy",
    #                 None,
    #             ),
    #             (
    #                 "precision",
    #                 0,
    #             ),
    #             (
    #                 "recall",
    #                 0,
    #             ),
    #             (
    #                 "f1_score",
    #                 0,
    #             ),
    #         ],
    #         ids=["accuracy", "precision", "recall", "f1_score"],
    #     )
    #     def test_should_assert_that_is_fitted_is_set_correctly_and_check_return_type_for_fnns(
    #         self,
    #         metric: Literal["accuracy", "precision", "recall", "f1_score"],
    #         positive_class: Any,
    #         device: Device,
    #     ) -> None:
    #         configure_test_with_device(device)
    #
    #         model = NeuralNetworkClassifier(InputConversionTable(), [ForwardLayer(Choice(2, 4)), ForwardLayer(1)])
    #         assert not model.is_fitted
    #         fitted_model = model.fit_by_exhaustive_search(
    #             Table.from_dict({"a": [1, 2, 3, 4], "b": [0, 1, 0, 1]}).to_tabular_dataset("b"),
    #             optimization_metric=metric,
    #             positive_class=positive_class,
    #         )
    #         assert fitted_model.is_fitted
    #         assert isinstance(fitted_model, NeuralNetworkClassifier)
    #
    #     @pytest.mark.parametrize(
    #         ("metric", "positive_class"),
    #         [
    #             (
    #                 "accuracy",
    #                 None,
    #             ),
    #             (
    #                 "precision",
    #                 0,
    #             ),
    #             (
    #                 "recall",
    #                 0,
    #             ),
    #             (
    #                 "f1_score",
    #                 0,
    #             ),
    #         ],
    #         ids=["accuracy", "precision", "recall", "f1_score"],
    #     )
    #     def test_should_assert_that_is_fitted_is_set_correctly_and_check_return_type_for_rnns(
    #         self,
    #         metric: Literal["accuracy", "precision", "recall", "f1_score"],
    #         positive_class: Any,
    #         device: Device,
    #     ) -> None:
    #         configure_test_with_device(device)
    #
    #         # Create a DataFrame
    #         _inflation_path = "_datas/US_Inflation_rates.csv"
    #         table = Table.from_csv_file(path=resolve_resource_path(_inflation_path))
    #         rs = RangeScaler(column_names="value")
    #         _, table = rs.fit_and_transform(table)
    #         train_table = table.to_time_series_dataset(
    #             "value",
    #             window_size=7,
    #             forecast_horizon=12,
    #             continuous=False,
    #             extra_names=["date"],
    #         )
    #         model = NeuralNetworkClassifier(
    #             InputConversionTimeSeries(),
    #             [ForwardLayer(neuron_count=Choice(128, 256)), GRULayer(128), LSTMLayer(neuron_count=1)],
    #         )
    #         assert not model.is_fitted
    #
    #         fitted_model = model.fit_by_exhaustive_search(
    #             train_table,
    #             optimization_metric=metric,
    #             positive_class=positive_class,
    #             epoch_count=2,
    #         )
    #
    #         assert fitted_model.is_fitted
    #         assert isinstance(fitted_model, NeuralNetworkClassifier)
    #
    #     @pytest.mark.parametrize(
    #         ("metric", "positive_class"),
    #         [
    #             (
    #                 "accuracy",
    #                 None,
    #             ),
    #             (
    #                 "precision",
    #                 0,
    #             ),
    #             (
    #                 "recall",
    #                 0,
    #             ),
    #             (
    #                 "f1_score",
    #                 0,
    #             ),
    #         ],
    #         ids=["accuracy", "precision", "recall", "f1_score"],
    #     )
    #     def test_should_assert_that_is_fitted_is_set_correctly_and_check_return_type_for_cnns_column_output(
    #         self,
    #         metric: Literal["accuracy", "precision", "recall", "f1_score"],
    #         positive_class: Any,
    #         device: Device,
    #     ) -> None:
    #         configure_test_with_device(device)
    #
    #         image_list, filenames = ImageList.from_files(resolve_resource_path(images_all()), return_filenames=True)
    #         image_list = image_list.resize(20, 20)
    #         classes = []
    #         for filename in filenames:
    #             groups = re.search(r"(.*)[\\/](.*)\.", filename)
    #             if groups is not None:
    #                 classes.append(groups.group(2))
    #         image_classes = Table({"class": classes})
    #         label_encoder = LabelEncoder()
    #         (_, image_classes_label_encoded) = label_encoder.fit_and_transform(image_classes)
    #         image_dataset = ImageDataset(image_list, image_classes_label_encoded.get_column("class"))
    #         num_of_classes: int = image_dataset.output_size if isinstance(image_dataset.output_size, int) else 0
    #         layers = [
    #             Convolutional2DLayer(1, 2),
    #             MaxPooling2DLayer(10),
    #             FlattenLayer(),
    #             ForwardLayer(Choice(10, 20)),
    #             ForwardLayer(num_of_classes),
    #         ]
    #         model = NeuralNetworkClassifier(
    #             InputConversionImageToColumn(image_dataset.input_size),
    #             layers,
    #         )
    #         assert not model.is_fitted
    #         fitted_model = model.fit_by_exhaustive_search(
    #             image_dataset,
    #             epoch_count=2,
    #             optimization_metric=metric,
    #             positive_class=positive_class,
    #         )
    #
    #         assert fitted_model.is_fitted
    #         assert isinstance(fitted_model, NeuralNetworkClassifier)
    #
    #     @pytest.mark.parametrize(
    #         ("metric", "positive_class"),
    #         [
    #             (
    #                 "accuracy",
    #                 None,
    #             ),
    #             (
    #                 "precision",
    #                 0,
    #             ),
    #             (
    #                 "recall",
    #                 0,
    #             ),
    #             (
    #                 "f1_score",
    #                 0,
    #             ),
    #         ],
    #         ids=["accuracy", "precision", "recall", "f1_score"],
    #     )
    #     def test_should_assert_that_is_fitted_is_set_correctly_and_check_return_type_for_cnns_table_output(
    #         self,
    #         metric: Literal["accuracy", "precision", "recall", "f1_score"],
    #         positive_class: Any,
    #         device: Device,
    #     ) -> None:
    #         configure_test_with_device(device)
    #
    #         image_list, filenames = ImageList.from_files(resolve_resource_path(images_all()), return_filenames=True)
    #         image_list = image_list.resize(20, 20)
    #         classes = []
    #         for filename in filenames:
    #             groups = re.search(r"(.*)[\\/](.*)\.", filename)
    #             if groups is not None:
    #                 classes.append(groups.group(2))
    #         image_classes = Table({"class": classes})
    #         one_hot_encoder = OneHotEncoder(column_names="class").fit(image_classes)
    #         image_classes_one_hot_encoded = one_hot_encoder.transform(image_classes)
    #         image_dataset = ImageDataset(image_list, image_classes_one_hot_encoded)
    #         num_of_classes: int = image_dataset.output_size if isinstance(image_dataset.output_size, int) else 0
    #         layers = [
    #             Convolutional2DLayer(1, 2),
    #             MaxPooling2DLayer(10),
    #             FlattenLayer(),
    #             ForwardLayer(Choice(10, 20)),
    #             ForwardLayer(num_of_classes),
    #         ]
    #         model = NeuralNetworkClassifier(
    #             InputConversionImageToTable(image_dataset.input_size),
    #             layers,
    #         )
    #         assert not model.is_fitted
    #         fitted_model = model.fit_by_exhaustive_search(
    #             image_dataset,
    #             epoch_count=2,
    #             optimization_metric=metric,
    #             positive_class=positive_class,
    #         )
    #
    #         assert fitted_model.is_fitted
    #         assert isinstance(fitted_model, NeuralNetworkClassifier)

    class TestPredict:
        @pytest.mark.parametrize(
            "batch_size",
            [
                1,
                2,
            ],
            ids=["one", "two"],
        )
        def test_should_raise_if_predict_function_returns_wrong_datatype(self, batch_size: int, device: Device) -> None:
            configure_test_with_device(device)
            fitted_model = NeuralNetworkClassifier(
                InputConversionTable(),
                [ForwardLayer(neuron_count=8), ForwardLayer(neuron_count=1)],
            ).fit(
                Table.from_dict({"a": [1, 0, 1, 0, 1, 0], "b": [0, 1, 0, 12, 3, 3]}).to_tabular_dataset("a"),
                batch_size=batch_size,
            )
            predictions = fitted_model.predict(Table.from_dict({"b": [1, 0]}))
            assert isinstance(predictions, TabularDataset)

        @pytest.mark.parametrize(
            "batch_size",
            [
                1,
                2,
            ],
            ids=["one", "two"],
        )
        def test_should_raise_if_predict_function_returns_wrong_datatype_for_multiclass_classification(
            self,
            batch_size: int,
            device: Device,
        ) -> None:
            configure_test_with_device(device)
            fitted_model = NeuralNetworkClassifier(
                InputConversionTable(),
                [ForwardLayer(neuron_count=8), ForwardLayer(neuron_count=3)],
            ).fit(
                Table.from_dict({"a": [0, 1, 2], "b": [0, 15, 51]}).to_tabular_dataset("a"),
                batch_size=batch_size,
            )
            NeuralNetworkClassifier(
                InputConversionTable(),
                [ForwardLayer(neuron_count=8), LSTMLayer(neuron_count=3)],
            ).fit(
                Table.from_dict({"a": [0, 1, 2], "b": [0, 15, 51]}).to_tabular_dataset("a"),
                batch_size=batch_size,
            )
            predictions = fitted_model.predict(Table.from_dict({"b": [1, 4, 124]}))
            assert isinstance(predictions, TabularDataset)

        def test_should_raise_if_model_has_not_been_fitted(self, device: Device) -> None:
            configure_test_with_device(device)
            with pytest.raises(NotFittedError, match="This model has not been fitted yet."):
                NeuralNetworkClassifier(
                    InputConversionTable(),
                    [ForwardLayer(neuron_count=1)],
                ).predict(
                    Table.from_dict({"a": [1]}),
                )

        def test_should_raise_if_test_features_mismatch(self, device: Device) -> None:
            configure_test_with_device(device)
            model = NeuralNetworkClassifier(
                InputConversionTable(),
                [ForwardLayer(neuron_count=1), ForwardLayer(neuron_count=3)],
            )
            model = model.fit(
                Table.from_dict({"a": [1, 0, 2], "b": [0, 15, 5]}).to_tabular_dataset("a"),
            )
            with pytest.raises(
                FeatureDataMismatchError,
                match="The features in the given table do not match with the specified feature columns names of the model.",
            ):
                model.predict(
                    Table.from_dict({"a": [1], "c": [2]}),
                )

    @pytest.mark.parametrize(
        ("input_conversion", "layers", "error_msg"),
        [
            (
                InputConversionTable(),
                [Convolutional2DLayer(1, 1)],
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionTable(),
                [ConvolutionalTranspose2DLayer(1, 1)],
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionTable(),
                [MaxPooling2DLayer(1)],
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionTable(),
                [AveragePooling2DLayer(1)],
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionTable(),
                [FlattenLayer()],
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImageToTable(ImageSize(1, 1, 1)),
                [Convolutional2DLayer(1, 1)],
                r"The output data would be 2-dimensional but the provided output conversion uses 1-dimensional data.",
            ),
            (
                InputConversionImageToColumn(ImageSize(1, 1, 1)),
                [Convolutional2DLayer(1, 1)],
                r"The output data would be 2-dimensional but the provided output conversion uses 1-dimensional data.",
            ),
            (
                InputConversionImageToTable(ImageSize(1, 1, 1)),
                [ConvolutionalTranspose2DLayer(1, 1)],
                r"The output data would be 2-dimensional but the provided output conversion uses 1-dimensional data.",
            ),
            (
                InputConversionImageToColumn(ImageSize(1, 1, 1)),
                [ConvolutionalTranspose2DLayer(1, 1)],
                r"The output data would be 2-dimensional but the provided output conversion uses 1-dimensional data.",
            ),
            (
                InputConversionImageToTable(ImageSize(1, 1, 1)),
                [MaxPooling2DLayer(1)],
                r"The output data would be 2-dimensional but the provided output conversion uses 1-dimensional data.",
            ),
            (
                InputConversionImageToColumn(ImageSize(1, 1, 1)),
                [MaxPooling2DLayer(1)],
                r"The output data would be 2-dimensional but the provided output conversion uses 1-dimensional data.",
            ),
            (
                InputConversionImageToTable(ImageSize(1, 1, 1)),
                [AveragePooling2DLayer(1)],
                r"The output data would be 2-dimensional but the provided output conversion uses 1-dimensional data.",
            ),
            (
                InputConversionImageToColumn(ImageSize(1, 1, 1)),
                [AveragePooling2DLayer(1)],
                r"The output data would be 2-dimensional but the provided output conversion uses 1-dimensional data.",
            ),
            (
                InputConversionImageToTable(ImageSize(1, 1, 1)),
                [FlattenLayer(), Convolutional2DLayer(1, 1)],
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImageToColumn(ImageSize(1, 1, 1)),
                [FlattenLayer(), Convolutional2DLayer(1, 1)],
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImageToTable(ImageSize(1, 1, 1)),
                [FlattenLayer(), ConvolutionalTranspose2DLayer(1, 1)],
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImageToColumn(ImageSize(1, 1, 1)),
                [FlattenLayer(), ConvolutionalTranspose2DLayer(1, 1)],
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImageToTable(ImageSize(1, 1, 1)),
                [FlattenLayer(), MaxPooling2DLayer(1)],
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImageToColumn(ImageSize(1, 1, 1)),
                [FlattenLayer(), MaxPooling2DLayer(1)],
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImageToTable(ImageSize(1, 1, 1)),
                [FlattenLayer(), AveragePooling2DLayer(1)],
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImageToColumn(ImageSize(1, 1, 1)),
                [FlattenLayer(), AveragePooling2DLayer(1)],
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImageToTable(ImageSize(1, 1, 1)),
                [FlattenLayer(), FlattenLayer()],
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImageToColumn(ImageSize(1, 1, 1)),
                [FlattenLayer(), FlattenLayer()],
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImageToTable(ImageSize(1, 1, 1)),
                [ForwardLayer(1)],
                r"The 2-dimensional data has to be flattened before using a 1-dimensional layer.",
            ),
            (
                InputConversionImageToColumn(ImageSize(1, 1, 1)),
                [ForwardLayer(1)],
                r"The 2-dimensional data has to be flattened before using a 1-dimensional layer.",
            ),
            (
                InputConversionImageToTable(ImageSize(1, 1, 1)),
                [],
                r"You need to provide at least one layer to a neural network.",
            ),
            (
                InputConversionImageToColumn(ImageSize(1, 1, 1)),
                [],
                r"You need to provide at least one layer to a neural network.",
            ),
            (
                InputConversionImageToColumn(VariableImageSize(1, 1, 1)),
                [FlattenLayer()],
                r"A NeuralNetworkClassifier cannot be used with a InputConversionImage that uses a VariableImageSize.",
            ),
            (
                InputConversionImageToImage(VariableImageSize(1, 1, 1)),
                [FlattenLayer()],
                r"A NeuralNetworkClassifier cannot be used with images as output.",
            ),
        ],
    )
    def test_should_raise_if_model_has_invalid_structure(
        self,
        input_conversion: InputConversion,
        layers: list[Layer],
        error_msg: str,
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        with pytest.raises(InvalidModelStructureError, match=error_msg):
            NeuralNetworkClassifier(input_conversion, layers)

    def test_should_be_pickleable(self, device: Device) -> None:
        configure_test_with_device(device)
        model = NeuralNetworkClassifier(
            InputConversionTable(),
            [
                ForwardLayer(1),
            ],
        )
        fitted_model = model.fit(
            Table(
                {
                    "a": [0],
                    "b": [0],
                },
            ).to_tabular_dataset("a"),
        )

        # Should not raise
        pickle.dumps(fitted_model)


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestRegressionModel:
    class TestFit:
        def test_should_return_input_size(self, device: Device) -> None:
            configure_test_with_device(device)
            model = NeuralNetworkRegressor(
                InputConversionTable(),
                [ForwardLayer(neuron_count=1)],
            ).fit(
                Table.from_dict({"a": [1], "b": [2]}).to_tabular_dataset("a"),
            )
            assert model.input_size == 1

        def test_should_raise_if_epoch_count_out_of_bounds(self, device: Device) -> None:
            invalid_epoch_count = 0
            configure_test_with_device(device)
            with pytest.raises(OutOfBoundsError):
                NeuralNetworkRegressor(
                    InputConversionTable(),
                    [ForwardLayer(neuron_count=1)],
                ).fit(
                    Table.from_dict({"a": [1], "b": [2]}).to_tabular_dataset("a"),
                    epoch_count=invalid_epoch_count,
                )

        def test_should_raise_if_batch_size_out_of_bounds(self, device: Device) -> None:
            invalid_batch_size = 0
            configure_test_with_device(device)
            with pytest.raises(OutOfBoundsError):
                NeuralNetworkRegressor(
                    InputConversionTable(),
                    [ForwardLayer(neuron_count=1)],
                ).fit(
                    Table.from_dict({"a": [1], "b": [2]}).to_tabular_dataset("a"),
                    batch_size=invalid_batch_size,
                )

        @pytest.mark.parametrize(
            "batch_size",
            [
                1,
                2,
            ],
            ids=["one", "two"],
        )
        def test_should_raise_if_fit_function_returns_wrong_datatype(self, batch_size: int, device: Device) -> None:
            configure_test_with_device(device)
            fitted_model = NeuralNetworkRegressor(
                InputConversionTable(),
                [ForwardLayer(neuron_count=1)],
            ).fit(
                Table.from_dict({"a": [1, 0, 1], "b": [2, 3, 4]}).to_tabular_dataset("a"),
                batch_size=batch_size,
            )
            assert isinstance(fitted_model, NeuralNetworkRegressor)

        def test_should_raise_when_fitting_with_choice(self, device: Device) -> None:
            configure_test_with_device(device)
            model = NeuralNetworkRegressor(InputConversionTable(), [ForwardLayer(Choice(1, 2))])
            with pytest.raises(FittingWithChoiceError):
                model.fit(Table.from_dict({"a": [1], "b": [0]}).to_tabular_dataset("a"))

        def test_should_raise_if_is_fitted_is_set_correctly(self, device: Device) -> None:
            configure_test_with_device(device)
            model = NeuralNetworkRegressor(
                InputConversionTable(),
                [ForwardLayer(neuron_count=1)],
            )
            assert not model.is_fitted
            model = model.fit(
                Table.from_dict({"a": [1], "b": [0]}).to_tabular_dataset("a"),
            )
            assert model.is_fitted

        def test_should_raise_if_train_features_mismatch(self, device: Device) -> None:
            configure_test_with_device(device)
            model = NeuralNetworkRegressor(
                InputConversionTable(),
                [ForwardLayer(neuron_count=1)],
            )
            trained_model = model.fit(
                Table.from_dict({"a": [1, 0, 2], "b": [0, 15, 5]}).to_tabular_dataset("b"),
            )
            with pytest.raises(
                FeatureDataMismatchError,
                match="The features in the given table do not match with the specified feature columns names of the model.",
            ):
                trained_model.fit(
                    Table.from_dict({"k": [1, 0, 2], "l": [0, 15, 5]}).to_tabular_dataset("l"),
                )

        @pytest.mark.parametrize(
            ("table", "reason"),
            [
                (
                    Table.from_dict({"a": [1, 2, 3], "b": [1, 2, None], "c": [0, 15, 5]}).to_tabular_dataset("c"),
                    re.escape("The given Fit Data is invalid:\nThe following Columns contain missing values: ['b']\n"),
                ),
                (
                    Table.from_dict({"a": ["a", "b", "c"], "b": [1, 2, 3], "c": [0, 15, 5]}).to_tabular_dataset("c"),
                    re.escape(
                        "The given Fit Data is invalid:\nThe following Columns contain non-numerical data: ['a']",
                    ),
                ),
                (
                    Table.from_dict({"a": ["a", "b", "c"], "b": [1, 2, None], "c": [0, 15, 5]}).to_tabular_dataset("c"),
                    re.escape(
                        "The given Fit Data is invalid:\nThe following Columns contain missing values: ['b']\nThe following Columns contain non-numerical data: ['a']",
                    ),
                ),
                (
                    Table.from_dict({"a": [1, 2, 3], "b": [1, 2, 3], "c": [0, None, 5]}).to_tabular_dataset("c"),
                    re.escape(
                        "The given Fit Data is invalid:\nThe following Columns contain missing values: ['c']\n",
                    ),
                ),
                (
                    Table.from_dict({"a": [1, 2, 3], "b": [1, 2, 3], "c": ["a", "b", "a"]}).to_tabular_dataset("c"),
                    re.escape(
                        "The given Fit Data is invalid:\nThe following Columns contain non-numerical data: ['c']",
                    ),
                ),
            ],
            ids=[
                "missing value feature",
                "non-numerical feature",
                "missing value and non-numerical features",
                "missing value target",
                "non-numerical target",
            ],
        )
        def test_should_catch_invalid_fit_data(self, device: Device, table: TabularDataset, reason: str) -> None:
            configure_test_with_device(device)
            model = NeuralNetworkRegressor(
                InputConversionTable(),
                [ForwardLayer(neuron_count=4), ForwardLayer(1)],
            )
            with pytest.raises(
                InvalidFitDataError,
                match=reason,
            ):
                model.fit(table)

        def test_should_raise_if_fit_doesnt_batch_callback(self, device: Device) -> None:
            configure_test_with_device(device)
            model = NeuralNetworkRegressor(
                InputConversionTable(),
                [ForwardLayer(neuron_count=1)],
            )

            class Test:
                self.was_called = False

                def cb(self, ind: int, loss: float) -> None:
                    if ind >= 0 and loss >= 0.0:
                        self.was_called = True

                def callback_was_called(self) -> bool:
                    return self.was_called

            obj = Test()
            model.fit(
                Table.from_dict({"a": [1], "b": [0]}).to_tabular_dataset("a"),
                callback_on_batch_completion=obj.cb,
            )

            assert obj.callback_was_called() is True

        def test_should_raise_if_fit_doesnt_epoch_callback(self, device: Device) -> None:
            configure_test_with_device(device)
            model = NeuralNetworkRegressor(
                InputConversionTable(),
                [ForwardLayer(neuron_count=1)],
            )

            class Test:
                self.was_called = False

                def cb(self, ind: int, loss: float) -> None:
                    if ind >= 0 and loss >= 0.0:
                        self.was_called = True

                def callback_was_called(self) -> bool:
                    return self.was_called

            obj = Test()
            model.fit(
                Table.from_dict({"a": [1], "b": [0]}).to_tabular_dataset("a"),
                callback_on_epoch_completion=obj.cb,
            )

            assert obj.callback_was_called() is True

    # TODO: Does not work if tensorflow with CUDA is used
    # class TestFitByExhaustiveSearch:
    #     def test_should_return_input_size(self, device: Device) -> None:
    #         configure_test_with_device(device)
    #         model = NeuralNetworkRegressor(
    #             InputConversionTable(),
    #             [ForwardLayer(neuron_count=Choice(2, 4)), ForwardLayer(1)],
    #         ).fit_by_exhaustive_search(
    #             Table.from_dict({"a": [1, 2, 3, 4], "b": [1.0, 2.0, 3.0, 4.0]}).to_tabular_dataset("b"),
    #             "mean_squared_error",
    #         )
    #         assert model.input_size == 1
    #
    #     def test_should_raise_if_epoch_count_out_of_bounds_when_fitting_by_exhaustive_search(
    #         self,
    #         device: Device,
    #     ) -> None:
    #         invalid_epoch_count = 0
    #         configure_test_with_device(device)
    #         with pytest.raises(OutOfBoundsError):
    #             NeuralNetworkRegressor(
    #                 InputConversionTable(),
    #                 [ForwardLayer(Choice(1, 3))],
    #             ).fit_by_exhaustive_search(
    #                 Table.from_dict({"a": [1], "b": [1.0]}).to_tabular_dataset("b"),
    #                 "mean_squared_error",
    #                 epoch_count=invalid_epoch_count,
    #             )
    #
    #     def test_should_raise_if_batch_size_out_of_bounds_when_fitting_by_exhaustive_search(
    #         self,
    #         device: Device,
    #     ) -> None:
    #         invalid_batch_size = 0
    #         configure_test_with_device(device)
    #         with pytest.raises(OutOfBoundsError):
    #             NeuralNetworkRegressor(
    #                 InputConversionTable(),
    #                 [ForwardLayer(neuron_count=Choice(1, 3))],
    #             ).fit_by_exhaustive_search(
    #                 Table.from_dict({"a": [1], "b": [1.0]}).to_tabular_dataset("b"),
    #                 "mean_squared_error",
    #                 batch_size=invalid_batch_size,
    #             )
    #
    #     def test_should_raise_when_fitting_by_exhaustive_search_without_choice(self, device: Device) -> None:
    #         configure_test_with_device(device)
    #         model = NeuralNetworkRegressor(InputConversionTable(), [ForwardLayer(1)])
    #         with pytest.raises(FittingWithoutChoiceError):
    #             model.fit_by_exhaustive_search(
    #                 Table.from_dict({"a": [1], "b": [1.0]}).to_tabular_dataset("b"),
    #                 "mean_squared_error",
    #             )
    #
    #     @pytest.mark.parametrize(
    #         "metric",
    #         [
    #             "mean_squared_error",
    #             "mean_absolute_error",
    #             "median_absolute_deviation",
    #             "coefficient_of_determination",
    #         ],
    #         ids=[
    #             "mean_squared_error",
    #             "mean_absolute_error",
    #             "median_absolute_deviation",
    #             "coefficient_of_determination",
    #         ],
    #     )
    #     def test_should_assert_that_is_fitted_is_set_correctly_and_check_return_type_for_fnns(
    #         self,
    #         metric: Literal[
    #             "mean_squared_error",
    #             "mean_absolute_error",
    #             "median_absolute_deviation",
    #             "coefficient_of_determination",
    #         ],
    #         device: Device,
    #     ) -> None:
    #         model = NeuralNetworkRegressor(InputConversionTable(), [ForwardLayer(Choice(2, 4)), ForwardLayer(1)])
    #         assert not model.is_fitted
    #         fitted_model = model.fit_by_exhaustive_search(
    #             Table.from_dict({"a": [1, 2, 3, 4], "b": [1.0, 2.0, 3.0, 4.0]}).to_tabular_dataset("b"),
    #             optimization_metric=metric,
    #         )
    #         device.type
    #         assert fitted_model.is_fitted
    #         assert isinstance(fitted_model, NeuralNetworkRegressor)
    #
    #     @pytest.mark.parametrize(
    #         "metric",
    #         [
    #             "mean_squared_error",
    #             "mean_absolute_error",
    #             "median_absolute_deviation",
    #             "coefficient_of_determination",
    #         ],
    #         ids=[
    #             "mean_squared_error",
    #             "mean_absolute_error",
    #             "median_absolute_deviation",
    #             "coefficient_of_determination",
    #         ],
    #     )
    #     def test_should_assert_that_is_fitted_is_set_correctly_and_check_return_type_for_rnns(
    #         self,
    #         metric: Literal[
    #             "mean_squared_error",
    #             "mean_absolute_error",
    #             "median_absolute_deviation",
    #             "coefficient_of_determination",
    #         ],
    #         device: Device,
    #     ) -> None:
    #         configure_test_with_device(device)
    #
    #         # Create a DataFrame
    #         _inflation_path = "_datas/US_Inflation_rates.csv"
    #         table = Table.from_csv_file(path=resolve_resource_path(_inflation_path))
    #         rs = RangeScaler(column_names="value")
    #         _, table = rs.fit_and_transform(table)
    #         train_table = table.to_time_series_dataset(
    #             "value",
    #             window_size=7,
    #             forecast_horizon=12,
    #             continuous=False,
    #             extra_names=["date"],
    #         )
    #         model = NeuralNetworkRegressor(
    #             InputConversionTimeSeries(),
    #             [ForwardLayer(neuron_count=Choice(128, 256)), GRULayer(128), LSTMLayer(neuron_count=1)],
    #         )
    #         assert not model.is_fitted
    #
    #         fitted_model = model.fit_by_exhaustive_search(train_table, optimization_metric=metric, epoch_count=2)
    #
    #         assert fitted_model.is_fitted
    #         assert isinstance(fitted_model, NeuralNetworkRegressor)
    #
    #     @pytest.mark.parametrize(
    #         "metric",
    #         [
    #             "mean_squared_error",
    #             "mean_absolute_error",
    #             "median_absolute_deviation",
    #             "coefficient_of_determination",
    #         ],
    #         ids=[
    #             "mean_squared_error",
    #             "mean_absolute_error",
    #             "median_absolute_deviation",
    #             "coefficient_of_determination",
    #         ],
    #     )
    #     def test_should_assert_that_is_fitted_is_set_correctly_and_check_return_type_for_rnns_continuous(
    #         self,
    #         metric: Literal[
    #             "mean_squared_error",
    #             "mean_absolute_error",
    #             "median_absolute_deviation",
    #             "coefficient_of_determination",
    #         ],
    #         device: Device,
    #     ) -> None:
    #         configure_test_with_device(device)
    #
    #         # Create a DataFrame
    #         _inflation_path = "_datas/US_Inflation_rates.csv"
    #         table = Table.from_csv_file(path=resolve_resource_path(_inflation_path))
    #         rs = RangeScaler(column_names="value")
    #         _, table = rs.fit_and_transform(table)
    #         train_table = table.to_time_series_dataset(
    #             "value",
    #             window_size=7,
    #             forecast_horizon=12,
    #             continuous=True,
    #             extra_names=["date"],
    #         )
    #         model = NeuralNetworkRegressor(
    #             InputConversionTimeSeries(),
    #             [ForwardLayer(neuron_count=Choice(128, 256)), LSTMLayer(neuron_count=12)],
    #         )
    #         assert not model.is_fitted
    #
    #         fitted_model = model.fit_by_exhaustive_search(train_table, optimization_metric=metric, epoch_count=2)
    #
    #         assert fitted_model.is_fitted
    #         assert isinstance(fitted_model, NeuralNetworkRegressor)

    class TestPredict:
        @pytest.mark.parametrize(
            "batch_size",
            [
                1,
                2,
            ],
            ids=["one", "two"],
        )
        def test_should_raise_if_predict_function_returns_wrong_datatype(self, batch_size: int, device: Device) -> None:
            configure_test_with_device(device)
            fitted_model = NeuralNetworkRegressor(
                InputConversionTable(),
                [ForwardLayer(neuron_count=1)],
            ).fit(
                Table.from_dict({"a": [1, 0, 1], "b": [2, 3, 4]}).to_tabular_dataset("a"),
                batch_size=batch_size,
            )
            predictions = fitted_model.predict(Table.from_dict({"b": [5, 6, 7]}))
            assert isinstance(predictions, TabularDataset)

        def test_should_raise_if_model_has_not_been_fitted(self, device: Device) -> None:
            configure_test_with_device(device)
            with pytest.raises(NotFittedError, match="This model has not been fitted yet."):
                NeuralNetworkRegressor(
                    InputConversionTable(),
                    [ForwardLayer(neuron_count=1)],
                ).predict(
                    Table.from_dict({"a": [1]}),
                )

        def test_should_raise_if_test_features_mismatch(self, device: Device) -> None:
            configure_test_with_device(device)
            model = NeuralNetworkRegressor(
                InputConversionTable(),
                [ForwardLayer(neuron_count=1)],
            )
            model = model.fit(
                Table.from_dict({"a": [1, 0, 2], "b": [0, 15, 5]}).to_tabular_dataset("a"),
            )
            with pytest.raises(
                FeatureDataMismatchError,
                match="The features in the given table do not match with the specified feature columns names of the model.",
            ):
                model.predict(
                    Table.from_dict({"a": [1], "c": [2]}),
                )

    @pytest.mark.parametrize(
        ("input_conversion", "layers", "error_msg"),
        [
            (
                InputConversionTable(),
                [Convolutional2DLayer(1, 1)],
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionTable(),
                [ConvolutionalTranspose2DLayer(1, 1)],
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionTable(),
                [MaxPooling2DLayer(1)],
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionTable(),
                [AveragePooling2DLayer(1)],
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionTable(),
                [FlattenLayer()],
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImageToImage(ImageSize(1, 1, 1)),
                [FlattenLayer()],
                r"The output data would be 1-dimensional but the provided output conversion uses 2-dimensional data.",
            ),
            (
                InputConversionImageToImage(ImageSize(1, 1, 1)),
                [FlattenLayer(), ForwardLayer(1)],
                r"The output data would be 1-dimensional but the provided output conversion uses 2-dimensional data.",
            ),
            (
                InputConversionImageToImage(ImageSize(1, 1, 1)),
                [FlattenLayer(), Convolutional2DLayer(1, 1)],
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImageToImage(ImageSize(1, 1, 1)),
                [FlattenLayer(), ConvolutionalTranspose2DLayer(1, 1)],
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImageToImage(ImageSize(1, 1, 1)),
                [FlattenLayer(), MaxPooling2DLayer(1)],
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImageToImage(ImageSize(1, 1, 1)),
                [FlattenLayer(), AveragePooling2DLayer(1)],
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImageToImage(ImageSize(1, 1, 1)),
                [FlattenLayer(), FlattenLayer()],
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImageToImage(ImageSize(1, 1, 1)),
                [ForwardLayer(1)],
                r"The 2-dimensional data has to be flattened before using a 1-dimensional layer.",
            ),
            (
                InputConversionImageToImage(ImageSize(1, 1, 1)),
                [],
                r"You need to provide at least one layer to a neural network.",
            ),
            (
                InputConversionImageToTable(ImageSize(1, 1, 1)),
                [FlattenLayer()],
                r"A NeuralNetworkRegressor cannot be used with images as input and 1-dimensional data as output.",
            ),
            (
                InputConversionImageToColumn(ImageSize(1, 1, 1)),
                [FlattenLayer()],
                r"A NeuralNetworkRegressor cannot be used with images as input and 1-dimensional data as output.",
            ),
        ],
    )
    def test_should_raise_if_model_has_invalid_structure(
        self,
        input_conversion: InputConversion,
        layers: list[Layer],
        error_msg: str,
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        with pytest.raises(InvalidModelStructureError, match=error_msg):
            NeuralNetworkRegressor(input_conversion, layers)

    def test_should_be_pickleable(self, device: Device) -> None:
        configure_test_with_device(device)
        model = NeuralNetworkRegressor(
            InputConversionTable(),
            [
                ForwardLayer(1),
            ],
        )
        fitted_model = model.fit(
            Table(
                {
                    "a": [0],
                    "b": [0],
                },
            ).to_tabular_dataset("a"),
        )

        # Should not raise
        pickle.dumps(fitted_model)
