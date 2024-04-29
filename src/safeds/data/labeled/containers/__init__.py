"""Classes that can store labeled data."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._image_dataset import ImageDataset

apipkg.initpkg(
    __name__,
    {
        "ImageDataset": "._image_dataset:ImageDataset",
    },
)

__all__ = [
    "ImageDataset",
]
