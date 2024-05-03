import pytest
from safeds.data.image.typing import ImageSize
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import (
    FeatureDataMismatchError,
    InputSizeError,
    InvalidModelStructureError,
    ModelNotFittedError,
    OutOfBoundsError,
)
from safeds.ml.nn import (
    AvgPooling2DLayer,
    Convolutional2DLayer,
    ConvolutionalTranspose2DLayer,
    FlattenLayer,
    ForwardLayer,
    InputConversionImage,
    InputConversionTable,
    MaxPooling2DLayer,
    NeuralNetworkClassifier,
    NeuralNetworkRegressor,
    OutputConversionImageToImage,
    OutputConversionImageToTable,
    OutputConversionTable,
)
from safeds.ml.nn import InputConversion, Layer, OutputConversion
from safeds.ml.nn._output_conversion_image import OutputConversionImageToColumn


class TestClassificationModel:
    @pytest.mark.parametrize(
        "epoch_size",
        [
            0,
        ],
        ids=["epoch_size_out_of_bounds"],
    )
    def test_should_raise_if_epoch_size_out_of_bounds(self, epoch_size: int) -> None:
        with pytest.raises(
            OutOfBoundsError,
            match=rf"epoch_size \(={epoch_size}\) is not inside \[1, \u221e\)\.",
        ):
            NeuralNetworkClassifier(
                InputConversionTable(["b"], "a"),
                [ForwardLayer(1, 1)],
                OutputConversionTable(),
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
    def test_should_raise_if_batch_size_out_of_bounds(self, batch_size: int) -> None:
        with pytest.raises(
            OutOfBoundsError,
            match=rf"batch_size \(={batch_size}\) is not inside \[1, \u221e\)\.",
        ):
            NeuralNetworkClassifier(
                InputConversionTable(["b"], "a"),
                [ForwardLayer(input_size=1, output_size=1)],
                OutputConversionTable(),
            ).fit(
                Table.from_dict({"a": [1], "b": [2]}).to_tabular_dataset("a"),
                batch_size=batch_size,
            )

    def test_should_raise_if_fit_function_returns_wrong_datatype(self) -> None:
        fitted_model = NeuralNetworkClassifier(
            InputConversionTable(["b"], "a"),
            [ForwardLayer(input_size=1, output_size=8), ForwardLayer(output_size=1)],
            OutputConversionTable(),
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
    def test_should_raise_if_predict_function_returns_wrong_datatype(self, batch_size: int) -> None:
        fitted_model = NeuralNetworkClassifier(
            InputConversionTable(["b"], "a"),
            [ForwardLayer(input_size=1, output_size=8), ForwardLayer(output_size=1)],
            OutputConversionTable(),
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
    ) -> None:
        fitted_model = NeuralNetworkClassifier(
            InputConversionTable(["b"], "a"),
            [ForwardLayer(input_size=1, output_size=8), ForwardLayer(output_size=3)],
            OutputConversionTable(),
        ).fit(
            Table.from_dict({"a": [0, 1, 2], "b": [0, 15, 51]}).to_tabular_dataset("a"),
            batch_size=batch_size,
        )
        predictions = fitted_model.predict(Table.from_dict({"b": [1, 4, 124]}))
        assert isinstance(predictions, TabularDataset)

    def test_should_raise_if_model_has_not_been_fitted(self) -> None:
        with pytest.raises(ModelNotFittedError, match="The model has not been fitted yet."):
            NeuralNetworkClassifier(
                InputConversionTable(["b"], "a"),
                [ForwardLayer(input_size=1, output_size=1)],
                OutputConversionTable(),
            ).predict(
                Table.from_dict({"a": [1]}),
            )

    def test_should_raise_if_is_fitted_is_set_correctly_for_binary_classification(self) -> None:
        model = NeuralNetworkClassifier(
            InputConversionTable(["b"], "a"),
            [ForwardLayer(input_size=1, output_size=1)],
            OutputConversionTable(),
        )
        assert not model.is_fitted
        model = model.fit(
            Table.from_dict({"a": [1], "b": [0]}).to_tabular_dataset("a"),
        )
        assert model.is_fitted

    def test_should_raise_if_is_fitted_is_set_correctly_for_multiclass_classification(self) -> None:
        model = NeuralNetworkClassifier(
            InputConversionTable(["b"], "a"),
            [ForwardLayer(input_size=1, output_size=1), ForwardLayer(output_size=3)],
            OutputConversionTable(),
        )
        assert not model.is_fitted
        model = model.fit(
            Table.from_dict({"a": [1, 0, 2], "b": [0, 15, 5]}).to_tabular_dataset("a"),
        )
        assert model.is_fitted

    def test_should_raise_if_test_features_mismatch(self) -> None:
        model = NeuralNetworkClassifier(
            InputConversionTable(["b"], "a"),
            [ForwardLayer(input_size=1, output_size=1), ForwardLayer(output_size=3)],
            OutputConversionTable(),
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

    def test_should_raise_if_train_features_mismatch(self) -> None:
        model = NeuralNetworkClassifier(
            InputConversionTable(["b"], "a"),
            [ForwardLayer(input_size=1, output_size=1), ForwardLayer(output_size=3)],
            OutputConversionTable(),
        )
        with pytest.raises(
            FeatureDataMismatchError,
            match="The features in the given table do not match with the specified feature columns names of the neural network.",
        ):
            model.fit(
                Table.from_dict({"a": [1, 0, 2], "b": [0, 15, 5]}).to_tabular_dataset("b"),
            )

    def test_should_raise_if_table_size_and_input_size_mismatch(self) -> None:
        model = NeuralNetworkClassifier(
            InputConversionTable(["b", "c"], "a"),
            [ForwardLayer(input_size=1, output_size=1), ForwardLayer(output_size=3)],
            OutputConversionTable(),
        )
        with pytest.raises(
            InputSizeError,
        ):
            model.fit(
                Table.from_dict({"a": [1, 0, 2], "b": [0, 15, 5], "c": [3, 33, 333]}).to_tabular_dataset("a"),
            )

    def test_should_raise_if_fit_doesnt_batch_callback(self) -> None:
        model = NeuralNetworkClassifier(
            InputConversionTable(["b"], "a"),
            [ForwardLayer(input_size=1, output_size=1)],
            OutputConversionTable(),
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

    def test_should_raise_if_fit_doesnt_epoch_callback(self) -> None:
        model = NeuralNetworkClassifier(
            InputConversionTable(["b"], "a"),
            [ForwardLayer(input_size=1, output_size=1)],
            OutputConversionTable(),
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
        ("input_conversion", "layers", "output_conversion", "error_msg"),
        [
            (
                InputConversionTable([], ""),
                [FlattenLayer()],
                OutputConversionImageToTable(),
                r"The defined model uses an output conversion for images but no input conversion for images.",
            ),
            (
                InputConversionTable([], ""),
                [FlattenLayer()],
                OutputConversionImageToColumn(),
                r"The defined model uses an output conversion for images but no input conversion for images.",
            ),
            (
                InputConversionTable([], ""),
                [FlattenLayer()],
                OutputConversionImageToImage(),
                r"A NeuralNetworkClassifier cannot be used with images as output.",
            ),
            (
                InputConversionTable([], ""),
                [Convolutional2DLayer(1, 1)],
                OutputConversionTable(),
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionTable([], ""),
                [ConvolutionalTranspose2DLayer(1, 1)],
                OutputConversionTable(),
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionTable([], ""),
                [MaxPooling2DLayer(1)],
                OutputConversionTable(),
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionTable([], ""),
                [AvgPooling2DLayer(1)],
                OutputConversionTable(),
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionTable([], ""),
                [FlattenLayer()],
                OutputConversionTable(),
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [FlattenLayer()],
                OutputConversionTable(),
                r"The defined model uses an input conversion for images but no output conversion for images.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [Convolutional2DLayer(1, 1)],
                OutputConversionImageToTable(),
                r"The output data would be 2-dimensional but the provided output conversion uses 1-dimensional data.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [Convolutional2DLayer(1, 1)],
                OutputConversionImageToColumn(),
                r"The output data would be 2-dimensional but the provided output conversion uses 1-dimensional data.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [ConvolutionalTranspose2DLayer(1, 1)],
                OutputConversionImageToTable(),
                r"The output data would be 2-dimensional but the provided output conversion uses 1-dimensional data.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [ConvolutionalTranspose2DLayer(1, 1)],
                OutputConversionImageToColumn(),
                r"The output data would be 2-dimensional but the provided output conversion uses 1-dimensional data.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [MaxPooling2DLayer(1)],
                OutputConversionImageToTable(),
                r"The output data would be 2-dimensional but the provided output conversion uses 1-dimensional data.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [MaxPooling2DLayer(1)],
                OutputConversionImageToColumn(),
                r"The output data would be 2-dimensional but the provided output conversion uses 1-dimensional data.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [AvgPooling2DLayer(1)],
                OutputConversionImageToTable(),
                r"The output data would be 2-dimensional but the provided output conversion uses 1-dimensional data.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [AvgPooling2DLayer(1)],
                OutputConversionImageToColumn(),
                r"The output data would be 2-dimensional but the provided output conversion uses 1-dimensional data.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [FlattenLayer(), Convolutional2DLayer(1, 1)],
                OutputConversionImageToTable(),
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [FlattenLayer(), Convolutional2DLayer(1, 1)],
                OutputConversionImageToColumn(),
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [FlattenLayer(), ConvolutionalTranspose2DLayer(1, 1)],
                OutputConversionImageToTable(),
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [FlattenLayer(), ConvolutionalTranspose2DLayer(1, 1)],
                OutputConversionImageToColumn(),
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [FlattenLayer(), MaxPooling2DLayer(1)],
                OutputConversionImageToTable(),
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [FlattenLayer(), MaxPooling2DLayer(1)],
                OutputConversionImageToColumn(),
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [FlattenLayer(), AvgPooling2DLayer(1)],
                OutputConversionImageToTable(),
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [FlattenLayer(), AvgPooling2DLayer(1)],
                OutputConversionImageToColumn(),
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [FlattenLayer(), FlattenLayer()],
                OutputConversionImageToTable(),
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [FlattenLayer(), FlattenLayer()],
                OutputConversionImageToColumn(),
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [ForwardLayer(1)],
                OutputConversionImageToTable(),
                r"The 2-dimensional data has to be flattened before using a 1-dimensional layer.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [ForwardLayer(1)],
                OutputConversionImageToColumn(),
                r"The 2-dimensional data has to be flattened before using a 1-dimensional layer.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [],
                OutputConversionImageToTable(),
                r"You need to provide at least one layer to a neural network.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [],
                OutputConversionImageToColumn(),
                r"You need to provide at least one layer to a neural network.",
            ),
        ],
    )
    def test_should_raise_if_model_has_invalid_structure(
        self,
        input_conversion: InputConversion,
        layers: list[Layer],
        output_conversion: OutputConversion,
        error_msg: str,
    ) -> None:
        with pytest.raises(InvalidModelStructureError, match=error_msg):
            NeuralNetworkClassifier(input_conversion, layers, output_conversion)


class TestRegressionModel:
    @pytest.mark.parametrize(
        "epoch_size",
        [
            0,
        ],
        ids=["epoch_size_out_of_bounds"],
    )
    def test_should_raise_if_epoch_size_out_of_bounds(self, epoch_size: int) -> None:
        with pytest.raises(
            OutOfBoundsError,
            match=rf"epoch_size \(={epoch_size}\) is not inside \[1, \u221e\)\.",
        ):
            NeuralNetworkRegressor(
                InputConversionTable(["b"], "a"),
                [ForwardLayer(input_size=1, output_size=1)],
                OutputConversionTable(),
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
    def test_should_raise_if_batch_size_out_of_bounds(self, batch_size: int) -> None:
        with pytest.raises(
            OutOfBoundsError,
            match=rf"batch_size \(={batch_size}\) is not inside \[1, \u221e\)\.",
        ):
            NeuralNetworkRegressor(
                InputConversionTable(["b"], "a"),
                [ForwardLayer(input_size=1, output_size=1)],
                OutputConversionTable(),
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
    def test_should_raise_if_fit_function_returns_wrong_datatype(self, batch_size: int) -> None:
        fitted_model = NeuralNetworkRegressor(
            InputConversionTable(["b"], "a"),
            [ForwardLayer(input_size=1, output_size=1)],
            OutputConversionTable(),
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
    def test_should_raise_if_predict_function_returns_wrong_datatype(self, batch_size: int) -> None:
        fitted_model = NeuralNetworkRegressor(
            InputConversionTable(["b"], "a"),
            [ForwardLayer(input_size=1, output_size=1)],
            OutputConversionTable(),
        ).fit(
            Table.from_dict({"a": [1, 0, 1], "b": [2, 3, 4]}).to_tabular_dataset("a"),
            batch_size=batch_size,
        )
        predictions = fitted_model.predict(Table.from_dict({"b": [5, 6, 7]}))
        assert isinstance(predictions, TabularDataset)

    def test_should_raise_if_model_has_not_been_fitted(self) -> None:
        with pytest.raises(ModelNotFittedError, match="The model has not been fitted yet."):
            NeuralNetworkRegressor(
                InputConversionTable(["b"], "a"),
                [ForwardLayer(input_size=1, output_size=1)],
                OutputConversionTable(),
            ).predict(
                Table.from_dict({"a": [1]}),
            )

    def test_should_raise_if_is_fitted_is_set_correctly(self) -> None:
        model = NeuralNetworkRegressor(
            InputConversionTable(["b"], "a"),
            [ForwardLayer(input_size=1, output_size=1)],
            OutputConversionTable(),
        )
        assert not model.is_fitted
        model = model.fit(
            Table.from_dict({"a": [1], "b": [0]}).to_tabular_dataset("a"),
        )
        assert model.is_fitted

    def test_should_raise_if_test_features_mismatch(self) -> None:
        model = NeuralNetworkRegressor(
            InputConversionTable(["b"], "a"),
            [ForwardLayer(input_size=1, output_size=1)],
            OutputConversionTable(),
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

    def test_should_raise_if_train_features_mismatch(self) -> None:
        model = NeuralNetworkRegressor(
            InputConversionTable(["b"], "a"),
            [ForwardLayer(input_size=1, output_size=1)],
            OutputConversionTable(),
        )
        with pytest.raises(
            FeatureDataMismatchError,
            match="The features in the given table do not match with the specified feature columns names of the neural network.",
        ):
            model.fit(
                Table.from_dict({"a": [1, 0, 2], "b": [0, 15, 5]}).to_tabular_dataset("b"),
            )

    def test_should_raise_if_table_size_and_input_size_mismatch(self) -> None:
        model = NeuralNetworkRegressor(
            InputConversionTable(["b", "c"], "a"),
            [ForwardLayer(input_size=1, output_size=1), ForwardLayer(output_size=3)],
            OutputConversionTable(),
        )
        with pytest.raises(
            InputSizeError,
        ):
            model.fit(
                Table.from_dict({"a": [1, 0, 2], "b": [0, 15, 5], "c": [3, 33, 333]}).to_tabular_dataset("a"),
            )

    def test_should_raise_if_fit_doesnt_batch_callback(self) -> None:
        model = NeuralNetworkRegressor(
            InputConversionTable(["b"], "a"),
            [ForwardLayer(input_size=1, output_size=1)],
            OutputConversionTable(),
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

    def test_should_raise_if_fit_doesnt_epoch_callback(self) -> None:
        model = NeuralNetworkRegressor(
            InputConversionTable(["b"], "a"),
            [ForwardLayer(input_size=1, output_size=1)],
            OutputConversionTable(),
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
        ("input_conversion", "layers", "output_conversion", "error_msg"),
        [
            (
                InputConversionTable([], ""),
                [FlattenLayer()],
                OutputConversionImageToImage(),
                r"The defined model uses an output conversion for images but no input conversion for images.",
            ),
            (
                InputConversionTable([], ""),
                [Convolutional2DLayer(1, 1)],
                OutputConversionTable(),
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionTable([], ""),
                [ConvolutionalTranspose2DLayer(1, 1)],
                OutputConversionTable(),
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionTable([], ""),
                [MaxPooling2DLayer(1)],
                OutputConversionTable(),
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionTable([], ""),
                [AvgPooling2DLayer(1)],
                OutputConversionTable(),
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionTable([], ""),
                [FlattenLayer()],
                OutputConversionTable(),
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [FlattenLayer()],
                OutputConversionTable(),
                r"The defined model uses an input conversion for images but no output conversion for images.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [FlattenLayer()],
                OutputConversionImageToImage(),
                r"The output data would be 1-dimensional but the provided output conversion uses 2-dimensional data.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [FlattenLayer(), ForwardLayer(1)],
                OutputConversionImageToImage(),
                r"The output data would be 1-dimensional but the provided output conversion uses 2-dimensional data.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [FlattenLayer(), Convolutional2DLayer(1, 1)],
                OutputConversionImageToImage(),
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [FlattenLayer(), ConvolutionalTranspose2DLayer(1, 1)],
                OutputConversionImageToImage(),
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [FlattenLayer(), MaxPooling2DLayer(1)],
                OutputConversionImageToImage(),
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [FlattenLayer(), AvgPooling2DLayer(1)],
                OutputConversionImageToImage(),
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [FlattenLayer(), FlattenLayer()],
                OutputConversionImageToImage(),
                r"You cannot use a 2-dimensional layer with 1-dimensional data.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [ForwardLayer(1)],
                OutputConversionImageToImage(),
                r"The 2-dimensional data has to be flattened before using a 1-dimensional layer.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [],
                OutputConversionImageToImage(),
                r"You need to provide at least one layer to a neural network.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [FlattenLayer()],
                OutputConversionImageToTable(),
                r"A NeuralNetworkRegressor cannot be used with images as input and 1-dimensional data as output.",
            ),
            (
                InputConversionImage(ImageSize(1, 1, 1)),
                [FlattenLayer()],
                OutputConversionImageToColumn(),
                r"A NeuralNetworkRegressor cannot be used with images as input and 1-dimensional data as output.",
            ),
        ],
    )
    def test_should_raise_if_model_has_invalid_structure(
        self,
        input_conversion: InputConversion,
        layers: list[Layer],
        output_conversion: OutputConversion,
        error_msg: str,
    ) -> None:
        with pytest.raises(InvalidModelStructureError, match=error_msg):
            NeuralNetworkRegressor(input_conversion, layers, output_conversion)
