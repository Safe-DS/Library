import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import FeatureDataMismatchError, InputSizeError, ModelNotFittedError, OutOfBoundsError
from safeds.ml.nn import (
    ForwardLayer,
    LSTMLayer,
    InputConversionTable,
    NeuralNetworkClassifier,
    NeuralNetworkRegressor,
    OutputConversionTable,
)


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
                InputConversionTable(),
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
                InputConversionTable(),
                [ForwardLayer(input_size=1, output_size=1)],
                OutputConversionTable(),
            ).fit(
                Table.from_dict({"a": [1], "b": [2]}).to_tabular_dataset("a"),
                batch_size=batch_size,
            )

    def test_should_raise_if_fit_function_returns_wrong_datatype(self) -> None:
        fitted_model = NeuralNetworkClassifier(
            InputConversionTable(),
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
            InputConversionTable(),
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
            InputConversionTable(),
            [ForwardLayer(input_size=1, output_size=8), ForwardLayer(output_size=3)],
            OutputConversionTable(),
        ).fit(
            Table.from_dict({"a": [0, 1, 2], "b": [0, 15, 51]}).to_tabular_dataset("a"),
            batch_size=batch_size,
        )
        NeuralNetworkClassifier(
            InputConversionTable(),
            [ForwardLayer(input_size=1, output_size=8), LSTMLayer(output_size=3)],
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
                InputConversionTable(),
                [ForwardLayer(input_size=1, output_size=1)],
                OutputConversionTable(),
            ).predict(
                Table.from_dict({"a": [1]}),
            )


    def test_should_raise_if_is_fitted_is_set_correctly_for_binary_classification(self) -> None:
        model = NeuralNetworkClassifier(
            InputConversionTable(),
            [ForwardLayer(input_size=1, output_size=1)],
            OutputConversionTable(),
        )
        model_2 = NeuralNetworkClassifier(
            InputConversionTable(),
            [LSTMLayer(input_size=1, output_size=1)],
            OutputConversionTable(),
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

    def test_should_raise_if_is_fitted_is_set_correctly_for_multiclass_classification(self) -> None:
        model = NeuralNetworkClassifier(
            InputConversionTable(),
            [ForwardLayer(input_size=1, output_size=1), ForwardLayer(output_size=3)],
            OutputConversionTable(),
        )
        model_2 = NeuralNetworkClassifier(
            InputConversionTable(),
            [ForwardLayer(input_size=1, output_size=1), LSTMLayer(output_size=3)],
            OutputConversionTable(),
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

    def test_should_raise_if_test_features_mismatch(self) -> None:
        model = NeuralNetworkClassifier(
            InputConversionTable(),
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

    def test_should_raise_if_table_size_and_input_size_mismatch(self) -> None:
        model = NeuralNetworkClassifier(
            InputConversionTable(),
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
            InputConversionTable(),
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
            InputConversionTable(),
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
                InputConversionTable(),
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
                InputConversionTable(),
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
            InputConversionTable(),
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
            InputConversionTable(),
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
                InputConversionTable(),
                [ForwardLayer(input_size=1, output_size=1)],
                OutputConversionTable(),
            ).predict(
                Table.from_dict({"a": [1]}),
            )

    def test_should_raise_if_is_fitted_is_set_correctly(self) -> None:
        model = NeuralNetworkRegressor(
            InputConversionTable(),
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
            InputConversionTable(),
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

    def test_should_raise_if_table_size_and_input_size_mismatch(self) -> None:
        model = NeuralNetworkRegressor(
            InputConversionTable(),
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
            InputConversionTable(),
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
            InputConversionTable(),
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
