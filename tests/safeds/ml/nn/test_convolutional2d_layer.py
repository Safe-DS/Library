import sys
from typing import Literal

import pytest
from safeds.data.image.typing import ImageSize
from safeds.ml.nn import Convolutional2DLayer, ConvolutionalTranspose2DLayer
from torch import nn


class TestConvolutional2DLayer:

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
            "out_channel",
            "out_width",
            "out_height",
        ),
        [
            (Convolutional2DLayer, nn.Conv2d, 30, 2, 2, 2, 30, 7, 12),
            (ConvolutionalTranspose2DLayer, nn.ConvTranspose2d, 30, 2, 2, 2, 30, 16, 36),
        ],
    )
    def test_should_create_pooling_layer(
        self,
        activation_function: Literal["sigmoid", "relu", "softmax"],
        activation_layer: type[nn.Module],
        conv_type: type[Convolutional2DLayer],
        torch_layer: type[nn.Module],
        output_channel: int,
        kernel_size: int,
        stride: int,
        padding: int,
        out_channel: int,
        out_width: int,
        out_height: int,
    ) -> None:
        layer = conv_type(output_channel, kernel_size, stride=stride, padding=padding)
        input_size = ImageSize(10, 20, 30, _ignore_invalid_channel=True)
        layer._set_input_size(input_size)
        assert layer.input_size == input_size
        assert layer.output_size == ImageSize(out_width, out_height, out_channel, _ignore_invalid_channel=True)
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
            (Convolutional2DLayer, 30, 2, 2, 2),
            (ConvolutionalTranspose2DLayer, 30, 2, 2, 2),
        ],
    )
    def test_should_raise_if_input_size_not_set(
        self,
        activation_function: Literal["sigmoid", "relu", "softmax"],
        conv_type: type[Convolutional2DLayer],
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
            (Convolutional2DLayer, 30, 2, 2, 2),
            (ConvolutionalTranspose2DLayer, 30, 2, 2, 2),
        ],
    )
    def test_should_raise_if_activation_function_not_set(
        self,
        conv_type: type[Convolutional2DLayer],
        output_channel: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        layer = conv_type(output_channel, kernel_size, stride=stride, padding=padding)
        input_size = ImageSize(10, 20, 30, _ignore_invalid_channel=True)
        layer._set_input_size(input_size)
        with pytest.raises(
            ValueError,
            match=r"The activation_function is not set. The internal layer can only be created when the activation_function is provided in the kwargs.",
        ):
            layer._get_internal_layer()

    @pytest.mark.parametrize(
        ("conv_type", "output_channel", "kernel_size", "stride", "padding"),
        [
            (Convolutional2DLayer, 30, 2, 2, 2),
            (ConvolutionalTranspose2DLayer, 30, 2, 2, 2),
        ],
    )
    def test_should_raise_if_unsupported_activation_function_is_set(
        self,
        conv_type: type[Convolutional2DLayer],
        output_channel: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        layer = conv_type(output_channel, kernel_size, stride=stride, padding=padding)
        input_size = ImageSize(10, 20, 30, _ignore_invalid_channel=True)
        layer._set_input_size(input_size)
        with pytest.raises(
            ValueError,
            match=r"The activation_function 'unknown' is not supported. Please choose one of the following: \['sigmoid', 'relu', 'softmax'\].",
        ):
            layer._get_internal_layer(activation_function="unknown")

    @pytest.mark.parametrize(
        ("conv_type", "output_channel", "kernel_size", "stride", "padding"),
        [
            (Convolutional2DLayer, 30, 2, 2, 2),
            (ConvolutionalTranspose2DLayer, 30, 2, 2, 2),
        ],
    )
    def test_should_raise_if_input_size_is_set_with_int(
        self,
        conv_type: type[Convolutional2DLayer],
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
            ("conv2dlayer1", "conv2dlayer2"),
            [
                (Convolutional2DLayer(1, 2), Convolutional2DLayer(1, 2)),
                (Convolutional2DLayer(1, 2, stride=3, padding=4), Convolutional2DLayer(1, 2, stride=3, padding=4)),
                (ConvolutionalTranspose2DLayer(1, 2), ConvolutionalTranspose2DLayer(1, 2)),
                (
                    ConvolutionalTranspose2DLayer(1, 2, stride=3, padding=4, output_padding=5),
                    ConvolutionalTranspose2DLayer(1, 2, stride=3, padding=4, output_padding=5),
                ),
            ],
        )
        def test_should_be_equal(self, conv2dlayer1: Convolutional2DLayer, conv2dlayer2: Convolutional2DLayer) -> None:
            assert conv2dlayer1 == conv2dlayer2
            assert conv2dlayer2 == conv2dlayer1

        @pytest.mark.parametrize(
            "conv2dlayer1",
            [
                Convolutional2DLayer(1, 2),
                Convolutional2DLayer(1, 2, stride=3, padding=4),
                ConvolutionalTranspose2DLayer(1, 2),
                ConvolutionalTranspose2DLayer(1, 2, stride=3, padding=4, output_padding=5),
            ],
        )
        @pytest.mark.parametrize(
            "conv2dlayer2",
            [
                Convolutional2DLayer(2, 2),
                Convolutional2DLayer(1, 1),
                Convolutional2DLayer(1, 2, stride=4, padding=4),
                Convolutional2DLayer(1, 2, stride=3, padding=3),
                ConvolutionalTranspose2DLayer(1, 1),
                ConvolutionalTranspose2DLayer(2, 2),
                ConvolutionalTranspose2DLayer(1, 2, stride=4, padding=4, output_padding=5),
                ConvolutionalTranspose2DLayer(1, 2, stride=3, padding=3, output_padding=5),
                ConvolutionalTranspose2DLayer(1, 2, stride=3, padding=4, output_padding=4),
            ],
        )
        def test_should_not_be_equal(
            self, conv2dlayer1: Convolutional2DLayer, conv2dlayer2: Convolutional2DLayer,
        ) -> None:
            assert conv2dlayer1 != conv2dlayer2
            assert conv2dlayer2 != conv2dlayer1

        def test_should_be_not_implemented(self) -> None:
            conv2dlayer = Convolutional2DLayer(1, 2)
            convtranspose2dlayer = ConvolutionalTranspose2DLayer(1, 2)
            assert conv2dlayer.__eq__(convtranspose2dlayer) is NotImplemented
            assert convtranspose2dlayer.__eq__(conv2dlayer) is NotImplemented

    class TestHash:

        @pytest.mark.parametrize(
            ("conv2dlayer1", "conv2dlayer2"),
            [
                (Convolutional2DLayer(1, 2), Convolutional2DLayer(1, 2)),
                (Convolutional2DLayer(1, 2, stride=3, padding=4), Convolutional2DLayer(1, 2, stride=3, padding=4)),
                (ConvolutionalTranspose2DLayer(1, 2), ConvolutionalTranspose2DLayer(1, 2)),
                (
                    ConvolutionalTranspose2DLayer(1, 2, stride=3, padding=4, output_padding=5),
                    ConvolutionalTranspose2DLayer(1, 2, stride=3, padding=4, output_padding=5),
                ),
            ],
        )
        def test_hash_should_be_equal(
            self, conv2dlayer1: Convolutional2DLayer, conv2dlayer2: Convolutional2DLayer,
        ) -> None:
            assert hash(conv2dlayer1) == hash(conv2dlayer2)

        @pytest.mark.parametrize(
            "conv2dlayer1",
            [
                Convolutional2DLayer(1, 2),
                Convolutional2DLayer(1, 2, stride=3, padding=4),
                ConvolutionalTranspose2DLayer(1, 2),
                ConvolutionalTranspose2DLayer(1, 2, stride=3, padding=4, output_padding=5),
            ],
        )
        @pytest.mark.parametrize(
            "conv2dlayer2",
            [
                Convolutional2DLayer(2, 2),
                Convolutional2DLayer(1, 1),
                Convolutional2DLayer(1, 2, stride=4, padding=4),
                Convolutional2DLayer(1, 2, stride=3, padding=3),
                ConvolutionalTranspose2DLayer(1, 1),
                ConvolutionalTranspose2DLayer(2, 2),
                ConvolutionalTranspose2DLayer(1, 2, stride=4, padding=4, output_padding=5),
                ConvolutionalTranspose2DLayer(1, 2, stride=3, padding=3, output_padding=5),
                ConvolutionalTranspose2DLayer(1, 2, stride=3, padding=4, output_padding=4),
            ],
        )
        def test_hash_should_not_be_equal(
            self, conv2dlayer1: Convolutional2DLayer, conv2dlayer2: Convolutional2DLayer,
        ) -> None:
            assert hash(conv2dlayer1) != hash(conv2dlayer2)

    class TestSizeOf:

        @pytest.mark.parametrize(
            "conv2dlayer",
            [
                Convolutional2DLayer(1, 2),
                Convolutional2DLayer(1, 2, stride=3, padding=4),
                ConvolutionalTranspose2DLayer(1, 2),
                ConvolutionalTranspose2DLayer(1, 2, stride=3, padding=4, output_padding=5),
            ],
        )
        def test_should_size_be_greater_than_normal_object(self, conv2dlayer: Convolutional2DLayer) -> None:
            assert sys.getsizeof(conv2dlayer) > sys.getsizeof(object())
