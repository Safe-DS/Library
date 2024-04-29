from torch import nn

from safeds.data.image.typing import ImageSize
from safeds.ml.nn import FlattenLayer


class TestFlattenLayer:

    def test_should_create_flatten_layer(self):
        layer = FlattenLayer()
        input_size = ImageSize(10, 20, 30, _ignore_invalid_channel=True)
        layer._set_input_size(input_size)
        assert layer.input_size == input_size
        assert layer.output_size == input_size.width * input_size.height * input_size.channel
        assert isinstance(next(next(layer._get_internal_layer().modules()).children()), nn.Flatten)
