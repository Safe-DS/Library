"""Classes that can store image data."""

from ._empty_image_list import _EmptyImageList
from ._image import Image
from ._image_list import ImageList
from ._multi_size_image_list import _MultiSizeImageList
from ._single_size_image_list import _SingleSizeImageList

__all__ = [
    "Image",
    "ImageList",
    "_SingleSizeImageList",
    "_MultiSizeImageList",
    "_EmptyImageList",
]
