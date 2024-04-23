"""Classes that can store image data."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._image import Image
    from ._image_list import ImageList
    from ._image_dataset import ImageDataset

apipkg.initpkg(
    __name__,
    {
        "Image": "._image:Image",
        "ImageList": "._image_list:ImageList",
        "ImageDataset": "._image_dataset:ImageDataset",
    },
)

__all__ = [
    "Image",
    "ImageList",
    "ImageDataset",
]
