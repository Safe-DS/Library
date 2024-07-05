import sys
from typing import Literal

import pytest
from safeds.data.tabular.containers import Table
from safeds.ml.nn.layers import Convolutional1DLayer
from torch import nn


class TestConvolutional1DLayer:
    @pytest.mark.parametrize(
        ("activation_function", "activation_layer"),
        [("sigmoid", nn.Sigmoid), ("relu", nn.ReLU), ("softmax", nn.Softmax)],
    )
    @pytest.mark.parametrize(
        (
            "conv_type",
            "torch_layer",
            "output_channel",
            "kernel_size",
            "stride",
            "padding",
            "out_size",
        ),
        [
            (Convolutional1DLayer, nn.Conv2d, 30, 2, 2, 2, 7),
        ],
    )
    def test_should_create_convolutional1d_layer(
        self,
        activation_function: Literal["sigmoid", "relu", "softmax"],
        activation_layer: type[nn.Module],
        conv_type: type[Convolutional1DLayer],
        torch_layer: type[nn.Module],
        output_channel: int,
        kernel_size: int,
        stride: int,
        padding: int,
        out_size: int,
    ) -> None:
        layer = conv_type(output_channel, kernel_size, stride=stride, padding=padding)
        input_size = 10
        layer._set_input_size(input_size)
        assert layer.input_size == input_size
        assert layer.output_size == out_size
        modules = list(next(layer._get_internal_layer(activation_function=activation_function).modules()).children())
        assert isinstance(modules[0], torch_layer)
        assert isinstance(modules[1], activation_layer)

    @pytest.mark.parametrize(
        "activation_function",
        [
            "sigmoid",
            "relu",
            "softmax",
        ],
    )
    @pytest.mark.parametrize(
        ("conv_type", "output_channel", "kernel_size", "stride", "padding"),
        [
            (Convolutional1DLayer, 30, 2, 2, 2),
        ],
    )
    def test_should_raise_if_input_size_not_set(
        self,
        activation_function: Literal["sigmoid", "relu", "softmax"],
        conv_type: type[Convolutional1DLayer],
        output_channel: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        layer = conv_type(output_channel, kernel_size, stride=stride, padding=padding)
        with pytest.raises(ValueError, match=r"The input_size is not yet set."):
            layer.input_size  # noqa: B018
        with pytest.raises(
            ValueError,
            match=r"The input_size is not yet set. The layer cannot compute the output_size if the input_size is not set.",
        ):
            layer.output_size  # noqa: B018
        with pytest.raises(
            ValueError,
            match=r"The input_size is not yet set. The internal layer can only be created when the input_size is set.",
        ):
            layer._get_internal_layer(activation_function=activation_function)

    @pytest.mark.parametrize(
        ("conv_type", "output_channel", "kernel_size", "stride", "padding"),
        [
            (Convolutional1DLayer, 30, 2, 2, 2),
        ],
    )
    def test_should_raise_if_activation_function_not_set(
        self,
        conv_type: type[Convolutional1DLayer],
        output_channel: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        layer = conv_type(output_channel, kernel_size, stride=stride, padding=padding)
        input_size = 10
        layer._set_input_size(input_size)
        with pytest.raises(
            ValueError,
            match=r"The activation_function is not set. The internal layer can only be created when the activation_function is provided in the kwargs.",
        ):
            layer._get_internal_layer()

    @pytest.mark.parametrize(
        ("conv_type", "output_channel", "kernel_size", "stride", "padding"),
        [
            (Convolutional1DLayer, 30, 2, 2, 2),
        ],
    )
    def test_should_raise_if_unsupported_activation_function_is_set(
        self,
        conv_type: type[Convolutional1DLayer],
        output_channel: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        layer = conv_type(output_channel, kernel_size, stride=stride, padding=padding)
        input_size = 10
        layer._set_input_size(input_size)
        with pytest.raises(
            ValueError,
            match=r"The activation_function 'unknown' is not supported. Please choose one of the following: \['sigmoid', 'relu', 'softmax'\].",
        ):
            layer._get_internal_layer(activation_function="unknown")

    @pytest.mark.parametrize(
        ("conv_type", "output_channel", "kernel_size", "stride", "padding"),
        [
            (Convolutional1DLayer, 30, 2, 2, 2),
        ],
    )
    def test_should_raise_if_input_size_is_set_with_int(
        self,
        conv_type: type[Convolutional1DLayer],
        output_channel: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        layer = conv_type(output_channel, kernel_size, stride=stride, padding=padding)
        with pytest.raises(TypeError, match=r"The input_size of a convolution layer has to be of type ImageSize."):
            layer._set_input_size(1)

    class TestEq:
        @pytest.mark.parametrize(
            ("conv1dlayer1", "conv1dlayer2"),
            [
                (Convolutional1DLayer(1, 2), Convolutional1DLayer(1, 2)),
                (Convolutional1DLayer(1, 2, stride=3, padding=4), Convolutional1DLayer(1, 2, stride=3, padding=4)),
            ],
        )
        def test_should_be_equal(self, conv1dlayer1: Convolutional1DLayer, conv1dlayer2: Convolutional1DLayer) -> None:
            assert conv1dlayer1 == conv1dlayer2
            assert conv1dlayer2 == conv1dlayer1

        @pytest.mark.parametrize(
            "conv1dlayer1",
            [
                Convolutional1DLayer(1, 2),
                Convolutional1DLayer(1, 2, stride=3, padding=4),
            ],
        )
        @pytest.mark.parametrize(
            "conv1dlayer2",
            [
                Convolutional1DLayer(2, 2),
                Convolutional1DLayer(1, 1),
                Convolutional1DLayer(1, 2, stride=4, padding=4),
                Convolutional1DLayer(1, 2, stride=3, padding=3),
            ],
        )
        def test_should_not_be_equal(
            self,
            conv2dlayer1: Convolutional1DLayer,
            conv2dlayer2: Convolutional1DLayer,
        ) -> None:
            assert conv2dlayer1 != conv2dlayer2
            assert conv2dlayer2 != conv2dlayer1

        def test_should_be_not_implemented(self) -> None:
            conv2dlayer = Convolutional1DLayer(1, 2)
            assert conv2dlayer.__eq__(Table()) is NotImplemented

    class TestHash:
        @pytest.mark.parametrize(
            ("conv2dlayer1", "conv2dlayer2"),
            [
                (Convolutional1DLayer(1, 2), Convolutional1DLayer(1, 2)),
                (Convolutional1DLayer(1, 2, stride=3, padding=4), Convolutional1DLayer(1, 2, stride=3, padding=4)),
            ],
        )
        def test_hash_should_be_equal(
            self,
            conv2dlayer1: Convolutional1DLayer,
            conv2dlayer2: Convolutional1DLayer,
        ) -> None:
            assert hash(conv2dlayer1) == hash(conv2dlayer2)

        @pytest.mark.parametrize(
            "conv2dlayer1",
            [
                Convolutional1DLayer(1, 2),
                Convolutional1DLayer(1, 2, stride=3, padding=4),
            ],
        )
        @pytest.mark.parametrize(
            "conv2dlayer2",
            [
                Convolutional1DLayer(2, 2),
                Convolutional1DLayer(1, 1),
                Convolutional1DLayer(1, 2, stride=4, padding=4),
                Convolutional1DLayer(1, 2, stride=3, padding=3),
            ],
        )
        def test_hash_should_not_be_equal(
            self,
            conv2dlayer1: Convolutional1DLayer,
            conv2dlayer2: Convolutional1DLayer,
        ) -> None:
            assert hash(conv2dlayer1) != hash(conv2dlayer2)

    class TestSizeOf:
        @pytest.mark.parametrize(
            "conv2dlayer",
            [
                Convolutional1DLayer(1, 2),
                Convolutional1DLayer(1, 2, stride=3, padding=4),
            ],
        )
        def test_should_size_be_greater_than_normal_object(self, conv2dlayer: Convolutional1DLayer) -> None:
            assert sys.getsizeof(conv2dlayer) > sys.getsizeof(object())
