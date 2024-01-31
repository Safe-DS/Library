import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import OutOfBoundsError
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
            ClassificationNeuralNetwork([FNNLayer(1, 1)]).fit(
                Table.from_dict({"a": [1], "b": [2]}).tag_columns("a"),
                batch_size=batch_size,
            )


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
            RegressionNeuralNetwork([FNNLayer(1, 1)]).train(
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
            RegressionNeuralNetwork([FNNLayer(1, 1)]).train(
                Table.from_dict({"a": [1], "b": [2]}).tag_columns("a"),
                batch_size=batch_size,
            )
