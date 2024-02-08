import pytest
from safeds.exceptions import OutOfBoundsError
from safeds.ml.nn import FNNLayer


@pytest.mark.parametrize(
    "input_size",
    [
        0,
    ],
    ids=["input_size_out_of_bounds"],
)
def test_should_raise_if_input_size_out_of_bounds(input_size: int) -> None:
    with pytest.raises(
        OutOfBoundsError,
        match=rf"input_size \(={input_size}\) is not inside \[1, \u221e\)\.",
    ):
        FNNLayer(input_size, 1)


@pytest.mark.parametrize(
    "output_size",
    [
        0,
    ],
    ids=["output_size_out_of_bounds"],
)
def test_should_raise_if_output_size_out_of_bounds(output_size: int) -> None:
    with pytest.raises(
        OutOfBoundsError,
        match=rf"output_size \(={output_size}\) is not inside \[1, \u221e\)\.",
    ):
        FNNLayer(1, output_size)

@pytest.mark.parametrize(
    "output_size",
    [
        (
            1
        ),
        (
            20
        ),
    ],
    ids=["one", "twenty"],
)
def test_should_raise_if_output_size_doesnt_match(output_size: int) -> None:
    assert FNNLayer(1, output_size).output_size == output_size
