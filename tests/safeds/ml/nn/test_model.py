import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import OutOfBoundsError
from safeds.ml.nn._fnn_layer import FNNLayer
from safeds.ml.nn._model import ClassificationModel, RegressionModel


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
            ClassificationModel([FNNLayer(1, 1)]).train(Table.from_dict({"a": [1], "b": [2]}).tag_columns("a"),
                                                        epoch_size=epoch_size)

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
            ClassificationModel([FNNLayer(1, 1)]).train(Table.from_dict({"a": [1], "b": [2]}).tag_columns("a"), batch_size=batch_size)


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
            RegressionModel([FNNLayer(1, 1)]).train(Table.from_dict({"a": [1], "b": [2]}).tag_columns("a"),
                                                    epoch_size=epoch_size)

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
            RegressionModel([FNNLayer(1, 1)]).train(Table.from_dict({"a": [1], "b": [2]}).tag_columns("a"),
                                                    batch_size=batch_size)
