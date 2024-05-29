import pytest
from safeds.data.image.typing import ImageSize
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import (
    FeatureDataMismatchError,
    InvalidModelStructureError,
    ModelNotFittedError,
    OutOfBoundsError,
)
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
from torch.types import Device

from tests.helpers import configure_test_with_device, get_devices, get_devices_ids


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestClassificationModel:
    def test_should_return_input_size(self, device: Device) -> None:
        configure_test_with_device(device)
        model = NeuralNetworkClassifier(
            InputConversionTable(),
            [ForwardLayer(neuron_count=1)],
        ).fit(
            Table.from_dict({"a": [1], "b": [2]}).to_tabular_dataset("a"),
        )

        assert model.input_size == 1

    @pytest.mark.parametrize(
        "epoch_size",
        [
            0,
        ],
        ids=["epoch_size_out_of_bounds"],
    )
    def test_should_raise_if_epoch_size_out_of_bounds(self, epoch_size: int, device: Device) -> None:
        configure_test_with_device(device)
        with pytest.raises(OutOfBoundsError):
            NeuralNetworkClassifier(
                InputConversionTable(),
                [ForwardLayer(1)],
            ).fit(
                Table.from_dict({"a": [1], "b": [2]}).to_tabular_dataset("a"),
                epoch_size=epoch_size,
            )

    @pytest.mark.parametrize(
        "batch_size",
        [
            0,
        ],
        ids=["batch_size_out_of_bounds"],
    )
    def test_should_raise_if_batch_size_out_of_bounds(self, batch_size: int, device: Device) -> None:
        configure_test_with_device(device)
        with pytest.raises(OutOfBoundsError):
            NeuralNetworkClassifier(
                InputConversionTable(),
                [ForwardLayer(neuron_count=1)],
            ).fit(
                Table.from_dict({"a": [1], "b": [2]}).to_tabular_dataset("a"),
                batch_size=batch_size,
            )

    def test_should_raise_if_fit_function_returns_wrong_datatype(self, device: Device) -> None:
        configure_test_with_device(device)
        fitted_model = NeuralNetworkClassifier(
            InputConversionTable(),
            [ForwardLayer(neuron_count=8), ForwardLayer(neuron_count=1)],
        ).fit(
            Table.from_dict({"a": [1], "b": [0]}).to_tabular_dataset("a"),
        )
        assert isinstance(fitted_model, NeuralNetworkClassifier)

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
        with pytest.raises(ModelNotFittedError, match="The model has not been fitted yet."):
            NeuralNetworkClassifier(
                InputConversionTable(),
                [ForwardLayer(neuron_count=1)],
            ).predict(
                Table.from_dict({"a": [1]}),
            )

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
            match="The features in the given table do not match with the specified feature columns names of the neural network.",
        ):
            model.predict(
                Table.from_dict({"a": [1], "c": [2]}),
            )

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
            match="The features in the given table do not match with the specified feature columns names of the neural network.",
        ):
            learned_model.fit(Table.from_dict({"k": [0.1, 0, 0.2], "l": [0, 0.15, 0.5]}).to_tabular_dataset("k"))

    # def test_should_raise_if_table_size_and_input_size_mismatch(self, device: Device) -> None:
    #     configure_test_with_device(device)
    #     model = NeuralNetworkClassifier(
    #         InputConversionTable(),
    #         [ForwardLayer(neuron_count=1), ForwardLayer(neuron_count=3)],
    #     )
    #     with pytest.raises(
    #         InputSizeError,
    #     ):
    #         model.fit(
    #             Table.from_dict({"a": [1, 0, 2], "b": [0, 15, 5], "c": [3, 33, 333]}).to_tabular_dataset("a"),
    #         )

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
        model.fit(Table.from_dict({"a": [1], "b": [0]}).to_tabular_dataset("a"), callback_on_batch_completion=obj.cb)

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
        model.fit(Table.from_dict({"a": [1], "b": [0]}).to_tabular_dataset("a"), callback_on_epoch_completion=obj.cb)

        assert obj.callback_was_called() is True

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


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestRegressionModel:
    def test_should_return_input_size(self, device: Device) -> None:
        configure_test_with_device(device)
        model = NeuralNetworkRegressor(
            InputConversionTable(),
            [ForwardLayer(neuron_count=1)],
        ).fit(
            Table.from_dict({"a": [1], "b": [2]}).to_tabular_dataset("a"),
        )

        assert model.input_size == 1

    @pytest.mark.parametrize(
        "epoch_size",
        [
            0,
        ],
        ids=["epoch_size_out_of_bounds"],
    )
    def test_should_raise_if_epoch_size_out_of_bounds(self, epoch_size: int, device: Device) -> None:
        configure_test_with_device(device)
        with pytest.raises(OutOfBoundsError):
            NeuralNetworkRegressor(
                InputConversionTable(),
                [ForwardLayer(neuron_count=1)],
            ).fit(
                Table.from_dict({"a": [1], "b": [2]}).to_tabular_dataset("a"),
                epoch_size=epoch_size,
            )

    @pytest.mark.parametrize(
        "batch_size",
        [
            0,
        ],
        ids=["batch_size_out_of_bounds"],
    )
    def test_should_raise_if_batch_size_out_of_bounds(self, batch_size: int, device: Device) -> None:
        configure_test_with_device(device)
        with pytest.raises(OutOfBoundsError):
            NeuralNetworkRegressor(
                InputConversionTable(),
                [ForwardLayer(neuron_count=1)],
            ).fit(
                Table.from_dict({"a": [1], "b": [2]}).to_tabular_dataset("a"),
                batch_size=batch_size,
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
        with pytest.raises(ModelNotFittedError, match="The model has not been fitted yet."):
            NeuralNetworkRegressor(
                InputConversionTable(),
                [ForwardLayer(neuron_count=1)],
            ).predict(
                Table.from_dict({"a": [1]}),
            )

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
            match="The features in the given table do not match with the specified feature columns names of the neural network.",
        ):
            model.predict(
                Table.from_dict({"a": [1], "c": [2]}),
            )

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
            match="The features in the given table do not match with the specified feature columns names of the neural network.",
        ):
            trained_model.fit(
                Table.from_dict({"k": [1, 0, 2], "l": [0, 15, 5]}).to_tabular_dataset("l"),
            )

    # def test_should_raise_if_table_size_and_input_size_mismatch(self, device: Device) -> None:
    #     configure_test_with_device(device)
    #     model = NeuralNetworkRegressor(
    #         InputConversionTable(),
    #         [ForwardLayer(neuron_count=1), ForwardLayer(neuron_count=3)],
    #     )
    #     with pytest.raises(
    #         InputSizeError,
    #     ):
    #         model.fit(
    #             Table.from_dict({"a": [1, 0, 2], "b": [0, 15, 5], "c": [3, 33, 333]}).to_tabular_dataset("a"),
    #         )

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
        model.fit(Table.from_dict({"a": [1], "b": [0]}).to_tabular_dataset("a"), callback_on_batch_completion=obj.cb)

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
        model.fit(Table.from_dict({"a": [1], "b": [0]}).to_tabular_dataset("a"), callback_on_epoch_completion=obj.cb)

        assert obj.callback_was_called() is True

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
