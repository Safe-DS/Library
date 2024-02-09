"""Classes that can store image data."""

from ._image import Image
from ._image_set import ImageSet
from ._fixed_sized_image_set import _FixedSizedImageSet
from ._various_sized_image_set import _VariousSizedImageSet

__all__ = [
    "Image",
    "ImageSet",
    "_FixedSizedImageSet",
    "_VariousSizedImageSet",
]
