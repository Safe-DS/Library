from typing import Literal, Type

import pytest
from torch import nn

from safeds.data.image.typing import ImageSize
from safeds.ml.nn._pooling2d_layer import _Pooling2DLayer


class TestPooling2DLayer:

    @pytest.mark.parametrize(
        ("strategy", "torch_layer"),
        [
            ("max", nn.MaxPool2d),
            ("avg", nn.AvgPool2d),
        ],
    )
    def test_should_create_pooling_layer(self, strategy: Literal["max", "avg"], torch_layer: Type[nn.Module]) -> None:
        layer = _Pooling2DLayer(strategy, 2, stride=2, padding=2)
        input_size = ImageSize(10, 20, 30, _ignore_invalid_channel=True)
        layer._set_input_size(input_size)
        assert layer.input_size == input_size
        assert layer.output_size == ImageSize(7, 12, 30, _ignore_invalid_channel=True)
        assert isinstance(next(next(layer._get_internal_layer().modules()).children()), torch_layer)
