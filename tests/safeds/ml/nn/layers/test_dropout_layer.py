import sys

import pytest
from safeds.ml.nn.layers import DropoutLayer
from safeds.data.tabular.containers import Table
from safeds.exceptions import OutOfBoundsError
from torch import nn

class TestDropoutLayer:
    def test_should_create_dropout_layer(self) -> None:
        size = 10
        layer = DropoutLayer(0.5)
        layer._set_input_size(size)
        assert layer.input_size == size
        assert layer.output_size == size
        assert isinstance(next(next(layer._get_internal_layer().modules()).children()), nn.Dropout)

    def test_should_check_bounds(self) -> None:
        with pytest.raises(OutOfBoundsError, match=r"propability must be in \(0, 1\) but was 2."):
            DropoutLayer(2)
        with pytest.raises(OutOfBoundsError, match=r"propability must be in \(0, 1\) but was -1."):
            DropoutLayer(-1)

    def test_input_size_should_be_set(self) -> None:
        layer = DropoutLayer(0.5)
        with pytest.raises(ValueError, match=r"The input_size is not yet set."):
            layer.input_size
        
        with pytest.raises(ValueError, match=r"The input_size is not yet set."):
            layer.output_size
        
        with pytest.raises(ValueError, match=r"The input_size is not yet set."):
            layer._get_internal_layer()
        
        with pytest.raises(ValueError, match=r"The input_size is not yet set."):
            layer.__sizeof__()


    def test_propability_is_set(self) -> None:
        propability_to_set = 0.5
        layer = DropoutLayer(propability_to_set)
        assert layer.propability == propability_to_set

class TestEq:
    def test_should_be_equal(self) -> None:
        assert DropoutLayer(0.5) == DropoutLayer(0.5)

    def test_should_be_not_implemented(self) -> None:
        assert DropoutLayer(0.5).__eq__(Table()) is NotImplemented

class TestHash:
    def test_hash_should_be_equal(self) -> None:
        assert hash(DropoutLayer(0.5)) == hash(DropoutLayer(0.5))

class TestSizeOf:
    def test_should_size_be_greater_than_normal_object(self) -> None:
        layer = DropoutLayer(0.5)
        layer._set_input_size(10)
        assert sys.getsizeof(layer) > sys.getsizeof(object())
