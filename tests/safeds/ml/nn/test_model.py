import pytest

from safeds.data.tabular.containers import Table
from safeds.ml.nn._fnn_layer import fnn_layer
from safeds.ml.nn._model import Model


@pytest.mark.parametrize(
    ("epoch_size", "batch_size", "expected_error_message"),
    [
        (
            0,
            5,
            (
                r"The Number of Epochs must be at least 1"
            ),
        ),
        (
            5,
            0,
            (
                r"Batch Size must be at least 1"
            ),
        ),
    ],
    ids=["epoch_size_out_of_bounds", "batch_size_out_of_bounds"],
)
def test_should_raise_error(epoch_size: int, batch_size: int, expected_error_message: str) -> None:
    with pytest.raises(ValueError, match=expected_error_message):
        Model([fnn_layer(1, 1)]).train(Table.from_dict({"a": [1], "b": [2]}).tag_columns("a"), epoch_size, batch_size)
