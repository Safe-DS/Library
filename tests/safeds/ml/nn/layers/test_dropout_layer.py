import sys

import pytest
from safeds.data.image.typing import ImageSize
from safeds.data.tabular.containers import Table
from safeds.ml.nn.layers import DropoutLayer
from torch import nn
from safeds.exceptions import OutOfBoundsError


class TestDropoutLayer:
    def test_should_check_bounds(self) -> None:
        with pytest.raises(OutOfBoundsError, match="propability must be in \(0, 1\) but was 2."):
            DropoutLayer(2)
        with pytest.raises(OutOfBoundsError, match="propability must be in \(0, 1\) but was -1."):
            DropoutLayer(-1)

    def test_input_size_should_be_set(self) -> None:
        with pytest.raises(ValueError, match="The input_size is not yet set."):
            layer = DropoutLayer(0.5)
            layer.input_size()
        with pytest.raises(ValueError, match="The input_size is not yet set."):
            layer = DropoutLayer(0.5)
            layer.output_size()
        with pytest.raises(ValueError, match="The input_size is not yet set."):
            layer = DropoutLayer(0.5)
            layer._get_internal_layer()