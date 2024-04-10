import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import ModelNotFittedError, OutOfBoundsError, TestTrainDataMismatchError
from safeds.ml.nn import FNNLayer, NeuralNetworkClassifier, NeuralNetworkRegressor


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
            NeuralNetworkClassifier([FNNLayer(1, 1)]).fit(
                Table.from_dict({"a": [1], "b": [2]}).tag_columns("a"),
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
            NeuralNetworkClassifier([FNNLayer(input_size=1, output_size=1)]).fit(
                Table.from_dict({"a": [1], "b": [2]}).tag_columns("a"),
                batch_size=batch_size,
            )

    def test_should_raise_if_fit_function_returns_wrong_datatype(self) -> None:
        fitted_model = NeuralNetworkClassifier(
            [FNNLayer(input_size=1, output_size=8), FNNLayer(output_size=1)],
        ).fit(
            Table.from_dict({"a": [1], "b": [0]}).tag_columns("a"),
        )
        assert isinstance(fitted_model, NeuralNetworkClassifier)

    def test_should_raise_if_predict_function_returns_wrong_datatype(self) -> None:
        fitted_model = NeuralNetworkClassifier(
            [FNNLayer(input_size=1, output_size=8), FNNLayer(output_size=1)],
        ).fit(
            Table.from_dict({"a": [1, 0], "b": [0, 1]}).tag_columns("a"),
        )
        predictions = fitted_model.predict(Table.from_dict({"b": [1, 0]}))
        assert isinstance(predictions, TaggedTable)

    def test_should_raise_if_predict_function_returns_wrong_datatype_for_multiclass_classification(self) -> None:
        fitted_model = NeuralNetworkClassifier(
            [FNNLayer(input_size=1, output_size=8), FNNLayer(output_size=3)],
        ).fit(
            Table.from_dict({"a": [0, 1, 2], "b": [0, 15, 51]}).tag_columns("a"),
        )
        predictions = fitted_model.predict(Table.from_dict({"b": [1]}))
        assert isinstance(predictions, TaggedTable)

    def test_should_raise_if_model_has_not_been_fitted(self) -> None:
        with pytest.raises(ModelNotFittedError, match="The model has not been fitted yet."):
            NeuralNetworkClassifier([FNNLayer(input_size=1, output_size=1)]).predict(
                Table.from_dict({"a": [1]}),
            )

    def test_should_raise_if_is_fitted_is_set_correctly_for_binary_classification(self) -> None:
        model = NeuralNetworkClassifier([FNNLayer(input_size=1, output_size=1)])
        assert not model.is_fitted
        model = model.fit(
            Table.from_dict({"a": [1], "b": [0]}).tag_columns("a"),
        )
        assert model.is_fitted

    def test_should_raise_if_is_fitted_is_set_correctly_for_multiclass_classification(self) -> None:
        model = NeuralNetworkClassifier([FNNLayer(input_size=1, output_size=1), FNNLayer(output_size=3)])
        assert not model.is_fitted
        model = model.fit(
            Table.from_dict({"a": [1, 0, 2], "b": [0, 15, 5]}).tag_columns("a"),
        )
        assert model.is_fitted

    def test_should_raise_if__test_and_train_data_mismatch(self) -> None:
        model = NeuralNetworkClassifier([FNNLayer(input_size=1, output_size=1), FNNLayer(output_size=3)])
        model = model.fit(
            Table.from_dict({"a": [1, 0, 2], "b": [0, 15, 5]}).tag_columns("a"),
        )
        with pytest.raises(
            TestTrainDataMismatchError,
            match="The column names in the test table do not match with the feature columns names of the training data.",
        ):
            model.predict(
                Table.from_dict({"a": [1], "c": [2]}),
            )

    def test_should_raise_if_fit_doesnt_batch_callback(self) -> None:
        model = NeuralNetworkClassifier([FNNLayer(input_size=1, output_size=1)])

        class Test:
            self.was_called = False

            def cb(self, ind: int, loss: float) -> None:
                if ind >= 0 and loss >= 0.0:
                    self.was_called = True

            def callback_was_called(self) -> bool:
                return self.was_called

        obj = Test()
        model.fit(Table.from_dict({"a": [1], "b": [0]}).tag_columns("a"), callback_on_batch_completion=obj.cb)

        assert obj.callback_was_called() is True

    def test_should_raise_if_fit_doesnt_epoch_callback(self) -> None:
        model = NeuralNetworkClassifier([FNNLayer(input_size=1, output_size=1)])

        class Test:
            self.was_called = False

            def cb(self, ind: int, loss: float) -> None:
                if ind >= 0 and loss >= 0.0:
                    self.was_called = True

            def callback_was_called(self) -> bool:
                return self.was_called

        obj = Test()
        model.fit(Table.from_dict({"a": [1], "b": [0]}).tag_columns("a"), callback_on_epoch_completion=obj.cb)

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
            NeuralNetworkRegressor([FNNLayer(input_size=1, output_size=1)]).fit(
                Table.from_dict({"a": [1], "b": [2]}).tag_columns("a"),
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
            NeuralNetworkRegressor([FNNLayer(input_size=1, output_size=1)]).fit(
                Table.from_dict({"a": [1], "b": [2]}).tag_columns("a"),
                batch_size=batch_size,
            )

    def test_should_raise_if_fit_function_returns_wrong_datatype(self) -> None:
        fitted_model = NeuralNetworkRegressor([FNNLayer(input_size=1, output_size=1)]).fit(
            Table.from_dict({"a": [1], "b": [2]}).tag_columns("a"),
        )
        assert isinstance(fitted_model, NeuralNetworkRegressor)

    def test_should_raise_if_predict_function_returns_wrong_datatype(self) -> None:
        fitted_model = NeuralNetworkRegressor([FNNLayer(input_size=1, output_size=1)]).fit(
            Table.from_dict({"a": [1], "b": [2]}).tag_columns("a"),
        )
        predictions = fitted_model.predict(Table.from_dict({"b": [1]}))
        assert isinstance(predictions, TaggedTable)

    def test_should_raise_if_model_has_not_been_fitted(self) -> None:
        with pytest.raises(ModelNotFittedError, match="The model has not been fitted yet."):
            NeuralNetworkRegressor([FNNLayer(input_size=1, output_size=1)]).predict(
                Table.from_dict({"a": [1]}),
            )

    def test_should_raise_if_is_fitted_is_set_correctly(self) -> None:
        model = NeuralNetworkRegressor([FNNLayer(input_size=1, output_size=1)])
        assert not model.is_fitted
        model = model.fit(
            Table.from_dict({"a": [1], "b": [0]}).tag_columns("a"),
        )
        assert model.is_fitted

    def test_should_raise_if__test_and_train_data_mismatch(self) -> None:
        model = NeuralNetworkRegressor([FNNLayer(input_size=1, output_size=1)])
        model = model.fit(
            Table.from_dict({"a": [1, 0, 2], "b": [0, 15, 5]}).tag_columns("a"),
        )
        with pytest.raises(
            TestTrainDataMismatchError,
            match="The column names in the test table do not match with the feature columns names of the training data.",
        ):
            model.predict(
                Table.from_dict({"a": [1], "c": [2]}),
            )

    def test_should_raise_if_fit_doesnt_batch_callback(self) -> None:
        model = NeuralNetworkRegressor([FNNLayer(input_size=1, output_size=1)])

        class Test:
            self.was_called = False

            def cb(self, ind: int, loss: float) -> None:
                if ind >= 0 and loss >= 0.0:
                    self.was_called = True

            def callback_was_called(self) -> bool:
                return self.was_called

        obj = Test()
        model.fit(Table.from_dict({"a": [1], "b": [0]}).tag_columns("a"), callback_on_batch_completion=obj.cb)

        assert obj.callback_was_called() is True

    def test_should_raise_if_fit_doesnt_epoch_callback(self) -> None:
        model = NeuralNetworkRegressor([FNNLayer(input_size=1, output_size=1)])

        class Test:
            self.was_called = False

            def cb(self, ind: int, loss: float) -> None:
                if ind >= 0 and loss >= 0.0:
                    self.was_called = True

            def callback_was_called(self) -> bool:
                return self.was_called

        obj = Test()
        model.fit(Table.from_dict({"a": [1], "b": [0]}).tag_columns("a"), callback_on_epoch_completion=obj.cb)

        assert obj.callback_was_called() is True
