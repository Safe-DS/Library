"""Classes that can store labeled data."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._image_dataset import ImageDataset
    from ._tabular_dataset import TabularDataset

apipkg.initpkg(
    __name__,
    {
        "ImageDataset": "._image_dataset:ImageDataset",
        "TabularDataset": "._tabular_dataset:TabularDataset",
    },
)

__all__ = [
    "ImageDataset",
    "TabularDataset",
]
