import sys
from typing import Literal

import pytest
from safeds.data.image.typing import ImageSize
from safeds.data.tabular.containers import Table
from safeds.ml.nn import AvgPooling2DLayer, MaxPooling2DLayer
from safeds.ml.nn._pooling2d_layer import _Pooling2DLayer
from torch import nn


class TestPooling2DLayer:

    @pytest.mark.parametrize(
        ("strategy", "torch_layer"),
        [
            ("max", nn.MaxPool2d),
            ("avg", nn.AvgPool2d),
        ],
    )
    def test_should_create_pooling_layer(self, strategy: Literal["max", "avg"], torch_layer: type[nn.Module]) -> None:
        layer = _Pooling2DLayer(strategy, 2, stride=2, padding=2)
        input_size = ImageSize(10, 20, 30, _ignore_invalid_channel=True)
        with pytest.raises(TypeError, match=r"The input_size of a pooling layer has to be of type ImageSize."):
            layer._set_input_size(1)
        layer._set_input_size(input_size)
        assert layer.input_size == input_size
        assert layer.output_size == ImageSize(7, 12, 30, _ignore_invalid_channel=True)
        assert isinstance(next(next(layer._get_internal_layer().modules()).children()), torch_layer)

    @pytest.mark.parametrize(
        "strategy",
        [
            "max",
            "avg",
        ],
    )
    def test_should_raise_if_input_size_not_set(self, strategy: Literal["max", "avg"]) -> None:
        layer = _Pooling2DLayer(strategy, 2, stride=2, padding=2)
        with pytest.raises(ValueError, match=r"The input_size is not yet set."):
            layer.input_size  # noqa: B018
        with pytest.raises(
            ValueError,
            match=r"The input_size is not yet set. The layer cannot compute the output_size if the input_size is not set.",
        ):
            layer.output_size  # noqa: B018

    @pytest.mark.parametrize(
        "strategy",
        [
            "max",
            "avg",
        ],
    )
    def test_should_raise_if_input_size_is_set_with_int(self, strategy: Literal["max", "avg"]) -> None:
        layer = _Pooling2DLayer(strategy, 2, stride=2, padding=2)
        with pytest.raises(TypeError, match=r"The input_size of a pooling layer has to be of type ImageSize."):
            layer._set_input_size(1)

    class TestEq:

        @pytest.mark.parametrize(
            ("pooling_2d_layer_1", "pooling_2d_layer_2"),
            [
                (MaxPooling2DLayer(2), MaxPooling2DLayer(2)),
                (MaxPooling2DLayer(2, stride=3, padding=4), MaxPooling2DLayer(2, stride=3, padding=4)),
                (AvgPooling2DLayer(2), AvgPooling2DLayer(2)),
                (AvgPooling2DLayer(2, stride=3, padding=4), AvgPooling2DLayer(2, stride=3, padding=4)),
            ],
        )
        def test_should_be_equal(
            self,
            pooling_2d_layer_1: _Pooling2DLayer,
            pooling_2d_layer_2: _Pooling2DLayer,
        ) -> None:
            assert pooling_2d_layer_1 == pooling_2d_layer_2

        @pytest.mark.parametrize(
            "pooling_2d_layer_1",
            [
                MaxPooling2DLayer(2),
                MaxPooling2DLayer(2, stride=3, padding=4),
                AvgPooling2DLayer(2),
                AvgPooling2DLayer(2, stride=3, padding=4),
            ],
        )
        @pytest.mark.parametrize(
            "pooling_2d_layer_2",
            [
                MaxPooling2DLayer(1),
                MaxPooling2DLayer(1, stride=3, padding=4),
                MaxPooling2DLayer(2, stride=1, padding=4),
                MaxPooling2DLayer(2, stride=3, padding=1),
                AvgPooling2DLayer(1),
                AvgPooling2DLayer(1, stride=3, padding=4),
                AvgPooling2DLayer(2, stride=1, padding=4),
                AvgPooling2DLayer(2, stride=3, padding=1),
            ],
        )
        def test_should_not_be_equal(
            self,
            pooling_2d_layer_1: _Pooling2DLayer,
            pooling_2d_layer_2: _Pooling2DLayer,
        ) -> None:
            assert pooling_2d_layer_1 != pooling_2d_layer_2

        def test_should_be_not_implemented(self) -> None:
            max_pooling_2d_layer = MaxPooling2DLayer(1)
            avg_pooling_2d_layer = AvgPooling2DLayer(1)
            other = Table()
            assert max_pooling_2d_layer.__eq__(other) is NotImplemented
            assert max_pooling_2d_layer.__eq__(avg_pooling_2d_layer) is NotImplemented
            assert avg_pooling_2d_layer.__eq__(other) is NotImplemented
            assert avg_pooling_2d_layer.__eq__(max_pooling_2d_layer) is NotImplemented

    class TestHash:

        @pytest.mark.parametrize(
            ("pooling_2d_layer_1", "pooling_2d_layer_2"),
            [
                (MaxPooling2DLayer(2), MaxPooling2DLayer(2)),
                (MaxPooling2DLayer(2, stride=3, padding=4), MaxPooling2DLayer(2, stride=3, padding=4)),
                (AvgPooling2DLayer(2), AvgPooling2DLayer(2)),
                (AvgPooling2DLayer(2, stride=3, padding=4), AvgPooling2DLayer(2, stride=3, padding=4)),
            ],
        )
        def test_hash_should_be_equal(
            self,
            pooling_2d_layer_1: _Pooling2DLayer,
            pooling_2d_layer_2: _Pooling2DLayer,
        ) -> None:
            assert hash(pooling_2d_layer_1) == hash(pooling_2d_layer_2)

        @pytest.mark.parametrize(
            "pooling_2d_layer_1",
            [
                MaxPooling2DLayer(2),
                MaxPooling2DLayer(2, stride=3, padding=4),
                AvgPooling2DLayer(2),
                AvgPooling2DLayer(2, stride=3, padding=4),
            ],
        )
        @pytest.mark.parametrize(
            "pooling_2d_layer_2",
            [
                MaxPooling2DLayer(1),
                MaxPooling2DLayer(1, stride=3, padding=4),
                MaxPooling2DLayer(2, stride=1, padding=4),
                MaxPooling2DLayer(2, stride=3, padding=1),
                AvgPooling2DLayer(1),
                AvgPooling2DLayer(1, stride=3, padding=4),
                AvgPooling2DLayer(2, stride=1, padding=4),
                AvgPooling2DLayer(2, stride=3, padding=1),
            ],
        )
        def test_hash_should_not_be_equal(
            self,
            pooling_2d_layer_1: _Pooling2DLayer,
            pooling_2d_layer_2: _Pooling2DLayer,
        ) -> None:
            assert hash(pooling_2d_layer_1) != hash(pooling_2d_layer_2)

    class TestSizeOf:

        @pytest.mark.parametrize(
            "pooling_2d_layer",
            [
                MaxPooling2DLayer(2),
                MaxPooling2DLayer(2, stride=3, padding=4),
                AvgPooling2DLayer(2),
                AvgPooling2DLayer(2, stride=3, padding=4),
            ],
        )
        def test_should_size_be_greater_than_normal_object(self, pooling_2d_layer: _Pooling2DLayer) -> None:
            assert sys.getsizeof(pooling_2d_layer) > sys.getsizeof(object())
