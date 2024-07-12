"""Types used to define neural networks and related attributes."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._model_image_size import ConstantImageSize, ModelImageSize, VariableImageSize
    from ._tensor_shape import TensorShape

apipkg.initpkg(
    __name__,
    {
        "ConstantImageSize": "._model_image_size:ConstantImageSize",
        "ModelImageSize": "._model_image_size:ModelImageSize",
        "TensorShape": "._tensor_shape:TensorShape",
        "VariableImageSize": "._model_image_size:VariableImageSize",
    },
)

__all__ = [
    "ConstantImageSize",
    "ModelImageSize",
    "TensorShape",
    "VariableImageSize",
]
