import sys
from typing import Any

import pytest
from safeds.data.image.typing import ImageSize
from safeds.exceptions import OutOfBoundsError
from safeds.ml.nn import ForwardLayer
from torch import nn


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
    ("activation_function", "expected_activation_function"),
    [
        ("sigmoid", nn.Sigmoid),
        ("relu", nn.ReLU),
        ("softmax", nn.Softmax),
        ("none", None),
    ],
    ids=["sigmoid", "relu", "softmax", "none"],
)
def test_should_accept_activation_function(activation_function: str, expected_activation_function: type | None) -> None:
    forward_layer = ForwardLayer(output_size=1, input_size=1)._get_internal_layer(
        activation_function=activation_function,
    )
    assert (
        forward_layer._fn is None
        if expected_activation_function is None
        else isinstance(forward_layer._fn, expected_activation_function)
    )


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
        ForwardLayer(output_size=1, input_size=1)._get_internal_layer(activation_function=activation_function)


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


def test_should_raise_if_input_size_is_set_with_image_size() -> None:
    layer = ForwardLayer(1)
    with pytest.raises(TypeError, match=r"The input_size of a forward layer has to be of type int."):
        layer._set_input_size(ImageSize(1, 2, 3))


def test_should_raise_if_activation_function_not_set() -> None:
    layer = ForwardLayer(1)
    with pytest.raises(
        ValueError,
        match=r"The activation_function is not set. The internal layer can only be created when the activation_function is provided in the kwargs.",
    ):
        layer._get_internal_layer()


@pytest.mark.parametrize(
    ("layer1", "layer2", "equal"),
    [
        (
            ForwardLayer(input_size=1, output_size=2),
            ForwardLayer(input_size=1, output_size=2),
            True,
        ),
        (
            ForwardLayer(input_size=1, output_size=2),
            ForwardLayer(input_size=2, output_size=1),
            False,
        ),
    ],
    ids=["equal", "not equal"],
)
def test_should_compare_forward_layers(layer1: ForwardLayer, layer2: ForwardLayer, equal: bool) -> None:
    assert (layer1.__eq__(layer2)) == equal


def test_should_assert_that_forward_layer_is_equal_to_itself() -> None:
    layer = ForwardLayer(input_size=1, output_size=1)
    assert layer.__eq__(layer)


@pytest.mark.parametrize(
    ("layer", "other"),
    [
        (ForwardLayer(input_size=1, output_size=1), None),
    ],
    ids=["ForwardLayer vs. None"],
)
def test_should_return_not_implemented_if_other_is_not_forward_layer(layer: ForwardLayer, other: Any) -> None:
    assert (layer.__eq__(other)) is NotImplemented


@pytest.mark.parametrize(
    ("layer1", "layer2"),
    [
        (
            ForwardLayer(input_size=1, output_size=2),
            ForwardLayer(input_size=1, output_size=2),
        ),
    ],
    ids=["equal"],
)
def test_should_assert_that_equal_forward_layers_have_equal_hash(layer1: ForwardLayer, layer2: ForwardLayer) -> None:
    assert layer1.__hash__() == layer2.__hash__()


@pytest.mark.parametrize(
    ("layer1", "layer2"),
    [
        (
            ForwardLayer(input_size=1, output_size=2),
            ForwardLayer(input_size=2, output_size=1),
        ),
    ],
    ids=["not equal"],
)
def test_should_assert_that_different_forward_layers_have_different_hash(
    layer1: ForwardLayer,
    layer2: ForwardLayer,
) -> None:
    assert layer1.__hash__() != layer2.__hash__()


@pytest.mark.parametrize(
    "layer",
    [
        ForwardLayer(input_size=1, output_size=1),
    ],
    ids=["one"],
)
def test_should_assert_that_layer_size_is_greater_than_normal_object(layer: ForwardLayer) -> None:
    assert sys.getsizeof(layer) > sys.getsizeof(object())
