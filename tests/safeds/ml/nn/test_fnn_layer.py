import pytest
from safeds.ml.nn._fnn_layer import FNNLayer


@pytest.mark.parametrize(
    ("input_size", "output_size", "expected_error_message"),
    [
        (
            0,
            5,
            (
                r"Input Size must be at least 1"
            ),
        ),
        (
            5,
            0,
            (
                r"Output Size must be at least 1"
            ),
        ),
    ],
    ids=["input_size_out_of_bounds", "output_size_out_of_bounds"],
)
def test_should_raise_error(input_size: int, output_size: int, expected_error_message: str) -> None:
    with pytest.raises(ValueError, match=expected_error_message):
        FNNLayer(input_size, output_size)
