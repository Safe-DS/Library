import sys

import pytest
from safeds.data.image.typing import ImageSize
from safeds.data.tabular.containers import Table
from safeds.ml.nn.layers import FlattenLayer
from torch import nn

from safeds.ml.nn.typing import VariableImageSize


class TestFlattenLayer:
    def test_should_create_flatten_layer(self) -> None:
        layer = FlattenLayer()
        input_size = ImageSize(10, 20, 30, _ignore_invalid_channel=True)
        layer._set_input_size(input_size)
        assert layer.input_size == input_size
        assert layer.output_size == input_size.width * input_size.height * input_size.channel
        assert isinstance(next(next(layer._get_internal_layer().modules()).children()), nn.Flatten)

    def test_should_raise_if_input_size_not_set(self) -> None:
        layer = FlattenLayer()
        with pytest.raises(ValueError, match=r"The input_size is not yet set."):
            layer.input_size  # noqa: B018
        with pytest.raises(
            ValueError,
            match=r"The input_size is not yet set. The layer cannot compute the output_size if the input_size is not set.",
        ):
            layer.output_size  # noqa: B018

    def test_should_raise_if_input_size_is_set_with_int(self) -> None:
        layer = FlattenLayer()
        with pytest.raises(TypeError, match=r"The input_size of a flatten layer has to be of type ImageSize."):
            layer._set_input_size(1)

    def test_should_raise_if_input_size_is_set_with_variable_image_size(self) -> None:
        layer = FlattenLayer()
        with pytest.raises(TypeError, match=r"The input_size of a flatten layer has to be a ConstantImageSize."):
            layer._set_input_size(VariableImageSize(1, 2, 3))

    class TestEq:
        def test_should_be_equal(self) -> None:
            assert FlattenLayer() == FlattenLayer()

        def test_should_be_not_implemented(self) -> None:
            assert FlattenLayer().__eq__(Table()) is NotImplemented

    class TestHash:
        def test_hash_should_be_equal(self) -> None:
            assert hash(FlattenLayer()) == hash(FlattenLayer())

    class TestSizeOf:
        def test_should_size_be_greater_than_normal_object(self) -> None:
            assert sys.getsizeof(FlattenLayer()) > sys.getsizeof(object())
