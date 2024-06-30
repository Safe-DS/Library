import sys

import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import OutOfBoundsError
from safeds.ml.nn.layers import DropoutLayer
from safeds.ml.nn.typing import ConstantImageSize
from torch import nn


class TestProbability:
    def test_should_be_accessible(self) -> None:
        probability = 0.5
        layer = DropoutLayer(probability)
        assert layer.probability == probability

    @pytest.mark.parametrize("probability", [-1, 2], ids=["too low", "too high"])
    def test_should_raise_if_out_of_bounds(self, probability: int) -> None:
        with pytest.raises(OutOfBoundsError):
            DropoutLayer(probability)


class TestDropoutLayer:
    def test_should_create_dropout_layer(self) -> None:
        size = 10
        layer = DropoutLayer(0.5)
        layer._set_input_size(size)
        assert layer.input_size == size
        assert layer.output_size == size
        assert isinstance(next(next(layer._get_internal_layer().modules()).children()), nn.Dropout)

    def test_input_size_should_be_set(self) -> None:
        layer = DropoutLayer(0.5)
        with pytest.raises(ValueError, match=r"The input_size is not yet set."):
            layer.input_size  # noqa: B018

        with pytest.raises(ValueError, match=r"The input_size is not yet set."):
            layer.output_size  # noqa: B018

        with pytest.raises(ValueError, match=r"The input_size is not yet set."):
            layer._get_internal_layer()

        with pytest.raises(ValueError, match=r"The input_size is not yet set."):
            layer.__sizeof__()


class TestEq:
    def test_should_be_equal(self) -> None:
        assert DropoutLayer(0.5) == DropoutLayer(0.5)

    def test_should_be_not_implemented(self) -> None:
        assert DropoutLayer(0.5).__eq__(Table()) is NotImplemented


class TestHash:
    def test_hash_should_be_equal(self) -> None:
        assert hash(DropoutLayer(0.5)) == hash(DropoutLayer(0.5))


class TestSizeOf:
    def test_should_int_size_be_greater_than_normal_object(self) -> None:
        layer = DropoutLayer(0.5)
        layer._set_input_size(10)
        assert sys.getsizeof(layer) > sys.getsizeof(object())

    def test_should_model_image_size_be_greater_than_normal_object(self) -> None:
        layer = DropoutLayer(0.5)
        layer._set_input_size(ConstantImageSize(1, 1, 1))
        assert sys.getsizeof(layer) > sys.getsizeof(object())
