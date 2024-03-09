import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import ModelNotFittedError, OutOfBoundsError
from safeds.ml.nn import ClassificationNeuralNetwork, FNNLayer, RegressionNeuralNetwork


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
            ClassificationNeuralNetwork([FNNLayer(1, 1)]).fit(
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
            ClassificationNeuralNetwork([FNNLayer(input_size=1, output_size=1)]).fit(
                Table.from_dict({"a": [1], "b": [2]}).tag_columns("a"),
                batch_size=batch_size,
            )

    def test_should_raise_if_fit_function_returns_wrong_datatype(self) -> None:
        fitted_model = ClassificationNeuralNetwork(
            [FNNLayer(input_size=1, output_size=8), FNNLayer(output_size=1)],
        ).fit(
            Table.from_dict({"a": [1], "b": [0]}).tag_columns("a"),
        )
        assert isinstance(fitted_model, ClassificationNeuralNetwork)

    def test_should_raise_if_predict_function_returns_wrong_datatype(self) -> None:
        fitted_model = ClassificationNeuralNetwork(
            [FNNLayer(input_size=1, output_size=8), FNNLayer(output_size=1)],
        ).fit(
            Table.from_dict({"a": [1], "b": [0]}).tag_columns("a"),
        )
        predictions = fitted_model.predict(Table.from_dict({"b": [1]}))
        assert isinstance(predictions, TaggedTable)

    def test_should_raise_if_model_has_not_been_fitted(self) -> None:
        with pytest.raises(ModelNotFittedError, match="The model has not been fitted yet."):
            ClassificationNeuralNetwork([FNNLayer(input_size=1, output_size=1)]).predict(
                Table.from_dict({"a": [1]}),
            )

    def test_should_raise_if_is_fitted_is_set_correctly(self) -> None:
        model = ClassificationNeuralNetwork([FNNLayer(input_size=1, output_size=1)])
        assert not model.is_fitted
        model = model.fit(
            Table.from_dict({"a": [1], "b": [0]}).tag_columns("a"),
        )
        assert model.is_fitted

    def test_should_raise_if_fit_doesnt_batch_callback(self) -> None:
        model = ClassificationNeuralNetwork([FNNLayer(input_size=1, output_size=1)])

        class Test:
            self.was_called = False

            def cb(self, ind: int, loss: float) -> None:
                if ind >= 0 and loss >= 0.0:
                    self.was_called = True

            def callback_was_called(self) -> bool:
                return self.was_called

        obj = Test()
        model.fit(
            Table.from_dict({"a": [1], "b": [0]}).tag_columns("a"), callback_on_batch_completion=obj.cb
        )

        assert obj.callback_was_called() is True

    def test_should_raise_if_fit_doesnt_epoch_callback(self) -> None:
        model = ClassificationNeuralNetwork([FNNLayer(input_size=1, output_size=1)])

        class Test:
            self.was_called = False

            def cb(self, ind: int, loss: float) -> None:
                if ind >= 0 and loss >= 0.0:
                    self.was_called = True

            def callback_was_called(self) -> bool:
                return self.was_called

        obj = Test()
        model.fit(
            Table.from_dict({"a": [1], "b": [0]}).tag_columns("a"), callback_on_epoch_completion=obj.cb
        )

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
            RegressionNeuralNetwork([FNNLayer(input_size=1, output_size=1)]).fit(
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
            RegressionNeuralNetwork([FNNLayer(input_size=1, output_size=1)]).fit(
                Table.from_dict({"a": [1], "b": [2]}).tag_columns("a"),
                batch_size=batch_size,
            )

    def test_should_raise_if_fit_function_returns_wrong_datatype(self) -> None:
        fitted_model = RegressionNeuralNetwork([FNNLayer(input_size=1, output_size=1)]).fit(
            Table.from_dict({"a": [1], "b": [2]}).tag_columns("a"),
        )
        assert isinstance(fitted_model, RegressionNeuralNetwork)

    def test_should_raise_if_predict_function_returns_wrong_datatype(self) -> None:
        fitted_model = RegressionNeuralNetwork([FNNLayer(input_size=1, output_size=1)]).fit(
            Table.from_dict({"a": [1], "b": [2]}).tag_columns("a"),
        )
        predictions = fitted_model.predict(Table.from_dict({"b": [1]}))
        assert isinstance(predictions, TaggedTable)

    def test_should_raise_if_model_has_not_been_fitted(self) -> None:
        with pytest.raises(ModelNotFittedError, match="The model has not been fitted yet."):
            RegressionNeuralNetwork([FNNLayer(input_size=1, output_size=1)]).predict(
                Table.from_dict({"a": [1]}),
            )

    def test_should_raise_if_is_fitted_is_set_correctly(self) -> None:
        model = RegressionNeuralNetwork([FNNLayer(input_size=1, output_size=1)])
        assert not model.is_fitted
        model = model.fit(
            Table.from_dict({"a": [1], "b": [0]}).tag_columns("a"),
        )
        assert model.is_fitted

    def test_should_raise_if_fit_doesnt_batch_callback(self) -> None:
        model = RegressionNeuralNetwork([FNNLayer(input_size=1, output_size=1)])

        class Test:
            self.was_called = False

            def cb(self, ind: int, loss: float) -> None:
                if ind >= 0 and loss >= 0.0:
                    self.was_called = True

            def callback_was_called(self) -> bool:
                return self.was_called

        obj = Test()
        model.fit(
            Table.from_dict({"a": [1], "b": [0]}).tag_columns("a"), callback_on_batch_completion=obj.cb
        )

        assert obj.callback_was_called() is True

    def test_should_raise_if_fit_doesnt_epoch_callback(self) -> None:
        model = RegressionNeuralNetwork([FNNLayer(input_size=1, output_size=1)])

        class Test:
            self.was_called = False

            def cb(self, ind: int, loss: float) -> None:
                if ind >= 0 and loss >= 0.0:
                    self.was_called = True

            def callback_was_called(self) -> bool:
                return self.was_called

        obj = Test()
        model.fit(
            Table.from_dict({"a": [1], "b": [0]}).tag_columns("a"), callback_on_epoch_completion=obj.cb
        )

        assert obj.callback_was_called() is True
