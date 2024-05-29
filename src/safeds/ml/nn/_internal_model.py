# The class must not be nested inside a function, since pickle cannot serialize local classes. Because of this, the
# slow import of torch must be on the global level. To still evaluate the torch import lazily, the class is moved to a
# separate file.

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn  # slow import

from safeds._config import _init_default_device
from safeds.ml.nn.converters._input_converter_image import _InputConversionImage
from safeds.ml.nn.layers import FlattenLayer, Layer
from safeds.ml.nn.layers._pooling2d_layer import _Pooling2DLayer

if TYPE_CHECKING:
    from safeds.ml.nn.converters import InputConversion
    from safeds.ml.nn.typing import ModelImageSize


# Use torch.compile once the following issues are resolved:
# - https://github.com/pytorch/pytorch/issues/120233 (Python 3.12 support)
# - https://github.com/triton-lang/triton/issues/1640 (Windows support)
class _InternalModel(nn.Module):
    def __init__(self, input_conversion: InputConversion, layers: list[Layer], is_for_classification: bool) -> None:
        super().__init__()

        _init_default_device()

        self._layer_list = layers
        internal_layers = []
        previous_output_size = input_conversion._data_size

        for layer in layers:
            if previous_output_size is not None:
                layer._set_input_size(previous_output_size)
            elif isinstance(input_conversion, _InputConversionImage):
                layer._set_input_size(input_conversion._data_size)
            if isinstance(layer, FlattenLayer | _Pooling2DLayer):
                internal_layers.append(layer._get_internal_layer())
            else:
                internal_layers.append(layer._get_internal_layer(activation_function="relu"))
            previous_output_size = layer.output_size

        if is_for_classification:
            internal_layers.pop()
            if isinstance(layers[-1].output_size, int) and layers[-1].output_size > 2:
                internal_layers.append(layers[-1]._get_internal_layer(activation_function="none"))
            else:
                internal_layers.append(layers[-1]._get_internal_layer(activation_function="sigmoid"))
        self._pytorch_layers = nn.Sequential(*internal_layers)

    @property
    def input_size(self) -> int | ModelImageSize:
        return self._layer_list[0].input_size

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._pytorch_layers:
            x = layer(x)
        return x
