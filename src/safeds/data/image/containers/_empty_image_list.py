from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from torch import Tensor

import xxhash

from safeds.data.image.containers import ImageList, Image, _SingleSizeImageList
from safeds.exceptions import IndexOutOfBoundsError


class _EmptyImageList(ImageList):

    _instance = None

    @staticmethod
    def _warn_empty_image_list() -> None:
        warnings.warn("You are using an empty ImageList. This method changes nothing if used on an empty ImageList.", UserWarning, stacklevel=2)

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def _create_image_list(images: list[Tensor], indices: list[int]) -> ImageList:
        raise NotImplementedError

    def clone(self) -> ImageList:
        return _EmptyImageList()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImageList):
            return NotImplemented
        return isinstance(other, _EmptyImageList)

    def __hash__(self) -> int:
        return xxhash.xxh3_64("_EmptyImageList").intdigest()

    def __sizeof__(self) -> int:
        return 0

    @property
    def number_of_images(self) -> int:
        return 0

    @property
    def widths(self) -> list[int]:
        return []

    @property
    def heights(self) -> list[int]:
        return []

    @property
    def channel(self) -> int:
        return NotImplemented

    @property
    def number_of_sizes(self) -> int:
        return 0

    def get_image(self, index: int) -> Image:
        raise IndexOutOfBoundsError(index)

    def index(self, *_) -> list[int]:
        return []

    def has_image(self, *_) -> bool:
        return False

    def to_jpeg_files(self, *_) -> None:
        warnings.warn("You are using an empty ImageList. No files will be saved.", UserWarning, stacklevel=2)
        return

    def to_png_files(self, *_) -> None:
        warnings.warn("You are using an empty ImageList. No files will be saved.", UserWarning, stacklevel=2)
        return

    def to_images(self, *_) -> list[Image]:
        return []

    def change_channel(self, *_) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        return _EmptyImageList()

    def _add_image_tensor(self, image_tensor: Tensor, *_) -> ImageList:
        return _SingleSizeImageList._create_image_list([image_tensor], [0])

    def add_images(self, images: list[Image] | ImageList) -> ImageList:
        return ImageList.from_images(images) if isinstance(images, list) else images.clone()

    def remove_image_by_index(self, *_) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        return _EmptyImageList()

    def remove_images_with_size(self,*_) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        return _EmptyImageList()

    def remove_duplicate_images(self) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        return _EmptyImageList()

    def shuffle_images(self) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        return _EmptyImageList()

    def resize(self, *_) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        return _EmptyImageList()

    def convert_to_grayscale(self) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        return _EmptyImageList()

    def crop(self, *_) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        return _EmptyImageList()

    def flip_vertically(self) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        return _EmptyImageList()

    def flip_horizontally(self) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        return _EmptyImageList()

    def adjust_brightness(self, *_) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        return _EmptyImageList()

    def add_noise(self, *_) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        return _EmptyImageList()

    def adjust_contrast(self, *_) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        return _EmptyImageList()

    def adjust_color_balance(self, *_) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        return _EmptyImageList()

    def blur(self, *_) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        return _EmptyImageList()

    def sharpen(self, *_) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        return _EmptyImageList()

    def invert_colors(self) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        return _EmptyImageList()

    def rotate_right(self) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        return _EmptyImageList()

    def rotate_left(self) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        return _EmptyImageList()

    def find_edges(self) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        return _EmptyImageList()

