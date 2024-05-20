"""Types used to define neural networks and related attributes."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._model_image_size import ConstantImageSize, ModelImageSize, VariableImageSize

apipkg.initpkg(
    __name__,
    {
        "ConstantImageSize": "._model_image_size:ConstantImageSize",
        "ModelImageSize": "._model_image_size:ModelImageSize",
        "VariableImageSize": "._model_image_size:VariableImageSize",
    },
)

__all__ = [
    "ConstantImageSize",
    "ModelImageSize",
    "VariableImageSize",
]
