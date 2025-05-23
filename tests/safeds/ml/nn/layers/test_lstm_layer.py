import sys
from typing import Any

import pytest
from torch import nn

from safeds.data.image.typing import ImageSize
from safeds.exceptions import OutOfBoundsError
from safeds.ml.hyperparameters import Choice
from safeds.ml.nn.layers import LSTMLayer

# TODO: Should be tested on a model, not a layer, since input size gets inferred
# @pytest.mark.parametrize(
#     "input_size",
#     [
#         1,
#         20,
#     ],
#     ids=["one", "twenty"],
# )
# def test_should_return_input_size(input_size: int) -> None:
#     assert LSTMLayer(neuron_count=1).input_size == input_size


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
    lstm_layer = LSTMLayer(neuron_count=1)
    lstm_layer._input_size = 1
    internal_layer = lstm_layer._get_internal_layer(
        activation_function=activation_function,
    )
    assert (
        internal_layer._fn is None
        if expected_activation_function is None
        else isinstance(internal_layer._fn, expected_activation_function)
    )


@pytest.mark.parametrize(
    "activation_function",
    [
        "unknown_string",
    ],
    ids=["unknown"],
)
def test_should_raise_if_unknown_activation_function_is_passed(activation_function: str) -> None:
    lstm_layer = LSTMLayer(neuron_count=1)
    lstm_layer._input_size = 1
    with pytest.raises(
        ValueError,
        match=rf"Unknown Activation Function: {activation_function}",
    ):
        lstm_layer._get_internal_layer(
            activation_function=activation_function,
        )


@pytest.mark.parametrize(
    "output_size",
    [
        0,
        Choice(0),
    ],
    ids=["invalid_int", "invalid_choice"],
)
def test_should_raise_if_output_size_out_of_bounds(output_size: int | Choice[int]) -> None:
    with pytest.raises(OutOfBoundsError):
        LSTMLayer(neuron_count=output_size)


@pytest.mark.parametrize(
    "output_size",
    [1, 20, Choice(1, 20)],
    ids=["one", "twenty", "choice"],
)
def test_should_raise_if_output_size_doesnt_match(output_size: int | Choice[int]) -> None:
    assert LSTMLayer(neuron_count=output_size).output_size == output_size


def test_should_raise_if_input_size_is_set_with_image_size() -> None:
    layer = LSTMLayer(1)
    with pytest.raises(TypeError, match=r"The input_size of a lstm layer has to be of type int."):
        layer._set_input_size(ImageSize(1, 2, 3))


def test_should_raise_if_activation_function_not_set() -> None:
    layer = LSTMLayer(1)
    with pytest.raises(
        ValueError,
        match=r"The activation_function is not set. The internal layer can only be created when the activation_function is provided in the kwargs.",
    ):
        layer._get_internal_layer()


@pytest.mark.parametrize(
    ("layer1", "layer2", "equal"),
    [
        (
            LSTMLayer(neuron_count=2),
            LSTMLayer(neuron_count=2),
            True,
        ),
        (
            LSTMLayer(neuron_count=2),
            LSTMLayer(neuron_count=1),
            False,
        ),
        (
            LSTMLayer(neuron_count=Choice(2)),
            LSTMLayer(neuron_count=Choice(2)),
            True,
        ),
        (
            LSTMLayer(neuron_count=Choice(2)),
            LSTMLayer(neuron_count=Choice(1)),
            False,
        ),
        (
            LSTMLayer(neuron_count=Choice(2)),
            LSTMLayer(neuron_count=2),
            False,
        ),
    ],
    ids=["equal", "not equal", "equal choices", "not equal choices", "choice and int"],
)
def test_should_compare_lstm_layers(layer1: LSTMLayer, layer2: LSTMLayer, equal: bool) -> None:
    assert (layer1.__eq__(layer2)) == equal


def test_should_assert_that_lstm_layer_is_equal_to_itself() -> None:
    layer = LSTMLayer(neuron_count=1)
    assert layer.__eq__(layer)


@pytest.mark.parametrize(
    ("layer", "other"),
    [
        (LSTMLayer(neuron_count=1), None),
    ],
    ids=["LSTMLayer vs. None"],
)
def test_should_return_not_implemented_if_other_is_not_lstm_layer(layer: LSTMLayer, other: Any) -> None:
    assert (layer.__eq__(other)) is NotImplemented


@pytest.mark.parametrize(
    ("layer1", "layer2"),
    [
        (
            LSTMLayer(neuron_count=2),
            LSTMLayer(neuron_count=2),
        ),
    ],
    ids=["equal"],
)
def test_should_assert_that_equal_lstm_layers_have_equal_hash(layer1: LSTMLayer, layer2: LSTMLayer) -> None:
    assert layer1.__hash__() == layer2.__hash__()


@pytest.mark.parametrize(
    ("layer1", "layer2"),
    [
        (
            LSTMLayer(neuron_count=2),
            LSTMLayer(neuron_count=1),
        ),
    ],
    ids=["not equal"],
)
def test_should_assert_that_different_lstm_layers_have_different_hash(
    layer1: LSTMLayer,
    layer2: LSTMLayer,
) -> None:
    assert layer1.__hash__() != layer2.__hash__()


@pytest.mark.parametrize(
    "layer",
    [
        LSTMLayer(neuron_count=1),
    ],
    ids=["one"],
)
def test_should_assert_that_layer_size_is_greater_than_normal_object(layer: LSTMLayer) -> None:
    assert sys.getsizeof(layer) > sys.getsizeof(object())


def test_should_get_all_possible_combinations_of_lstm_layer() -> None:
    layer = LSTMLayer(Choice(1, 2))
    possible_layers = layer._get_layers_for_all_choices()
    assert possible_layers[0] == LSTMLayer(1)
    assert possible_layers[1] == LSTMLayer(2)
