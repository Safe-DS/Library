import pytest
from safeds.exceptions import OutOfBoundsError
from safeds.ml.nn import ForwardLayer


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
        ForwardLayer(output_size=1, input_size=input_size)


@pytest.mark.parametrize(
    "input_size",
    [
        1,
        20,
    ],
    ids=["one", "twenty"],
)
def test_should_raise_if_input_size_doesnt_match(input_size: int) -> None:
    assert ForwardLayer(output_size=1, input_size=input_size).input_size == input_size


@pytest.mark.parametrize(
    "activation_function",
    [
        "unknown_string",
    ],
    ids=["unknown"],
)
def test_should_raise_if_unknown_activation_function_is_passed(activation_function: str) -> None:
    with pytest.raises(
        ValueError,
        match=rf"Unknown Activation Function: {activation_function}",
    ):
        ForwardLayer(output_size=1, input_size=1)._get_internal_layer(activation_function)


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
        ForwardLayer(output_size=output_size, input_size=1)


@pytest.mark.parametrize(
    "output_size",
    [
        1,
        20,
    ],
    ids=["one", "twenty"],
)
def test_should_raise_if_output_size_doesnt_match(output_size: int) -> None:
    assert ForwardLayer(output_size=output_size, input_size=1).output_size == output_size
