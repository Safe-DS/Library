"""Types used to define the attributes of image data."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._image_size import ImageSize

apipkg.initpkg(
    __name__,
    {
        "ImageSize": "._image_size:ImageSize",
    },
)

__all__ = [
    "ImageSize",
]
