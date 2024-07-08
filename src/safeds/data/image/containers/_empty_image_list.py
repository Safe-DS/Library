from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Self

from safeds._utils import _structural_hash
from safeds.data.image._utils._image_transformation_error_and_warning_checks import (
    _check_add_noise_errors,
    _check_adjust_brightness_errors_and_warnings,
    _check_adjust_color_balance_errors_and_warnings,
    _check_adjust_contrast_errors_and_warnings,
    _check_blur_errors_and_warnings,
    _check_crop_errors,
    _check_remove_images_with_size_errors,
    _check_resize_errors,
    _check_sharpen_errors_and_warnings,
)
from safeds.data.image.containers._image_list import ImageList
from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
from safeds.exceptions import IndexOutOfBoundsError

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

    from safeds.data.image.containers import Image
    from safeds.data.image.typing import ImageSize


class _EmptyImageList(ImageList):
    """
    An ImageList is a list of different images. It can hold different sizes of Images. The channel of all images is the same.

    This is the class for an empty ImageList.

    To create an `ImageList` call one of the following static methods:

    | Method                                                                        | Description                                              |
    | ----------------------------------------------------------------------------- | -------------------------------------------------------- |
    | [from_images][safeds.data.image.containers._image_list.ImageList.from_images] | Create an ImageList from a list of Images.               |
    | [from_files][safeds.data.image.containers._image_list.ImageList.from_files]   | Create an ImageList from a directory or a list of files. |
    """

    _instance = None

    @staticmethod
    def _warn_empty_image_list() -> None:
        """Warn if a transform method is used on an empty ImageList."""
        warnings.warn(
            "You are using an empty ImageList. This method changes nothing if used on an empty ImageList.",
            UserWarning,
            stacklevel=2,
        )

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def _create_image_list(images: list[Tensor], indices: list[int]) -> ImageList:
        raise NotImplementedError

    def _clone(self) -> ImageList:
        return _EmptyImageList()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImageList):
            return NotImplemented
        return isinstance(other, _EmptyImageList)

    def __hash__(self) -> int:
        return _structural_hash("_EmptyImageList")

    def __sizeof__(self) -> int:
        return 0

    @property
    def image_count(self) -> int:
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
    def sizes(self) -> list[ImageSize]:
        return []

    @property
    def size_count(self) -> int:
        return 0

    def get_image(self, index: int) -> Image:
        raise IndexOutOfBoundsError(index)

    def index(self, _image: Image) -> list[int]:
        return []

    def has_image(self, _image: Image) -> bool:
        return False

    def to_jpeg_files(self, _path: str | Path | list[str | Path]) -> None:
        warnings.warn("You are using an empty ImageList. No files will be saved.", UserWarning, stacklevel=2)

    def to_png_files(self, _path: str | Path | list[str | Path]) -> None:
        warnings.warn("You are using an empty ImageList. No files will be saved.", UserWarning, stacklevel=2)

    def to_images(self, _indices: list[int] | None = None) -> list[Image]:
        return []

    def change_channel(self, channel: int) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        if channel not in (1, 3, 4):
            raise ValueError(f"Channel {channel} is not a valid channel option. Use either 1, 3 or 4")
        return _EmptyImageList()

    def _add_image_tensor(self, image_tensor: Tensor, _index: int) -> ImageList:
        return _SingleSizeImageList._create_image_list([image_tensor], [0])

    def add_images(self, images: list[Image] | ImageList) -> ImageList:
        return ImageList.from_images(images) if isinstance(images, list) else images

    def remove_image_by_index(self, index: int | list[int]) -> ImageList:
        raise IndexOutOfBoundsError(index)

    def _remove_image_by_index_ignore_invalid(self, _index: int | list[int]) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        return _EmptyImageList()

    def remove_images_with_size(self, width: int, height: int) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        _check_remove_images_with_size_errors(width, height)
        return _EmptyImageList()

    def remove_duplicate_images(self) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        return _EmptyImageList()

    def shuffle_images(self) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        return _EmptyImageList()

    def resize(self, new_width: int, new_height: int) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        _check_resize_errors(new_width, new_height)
        return _EmptyImageList()

    def convert_to_grayscale(self) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        return _EmptyImageList()

    def crop(self, x: int, y: int, width: int, height: int) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        _check_crop_errors(x, y, width, height)
        return _EmptyImageList()

    def flip_vertically(self) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        return _EmptyImageList()

    def flip_horizontally(self) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        return _EmptyImageList()

    def adjust_brightness(self, factor: float) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        _check_adjust_brightness_errors_and_warnings(factor, plural=True)
        return _EmptyImageList()

    def add_noise(self, standard_deviation: float) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        _check_add_noise_errors(standard_deviation)
        return _EmptyImageList()

    def adjust_contrast(self, factor: float) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        _check_adjust_contrast_errors_and_warnings(factor, plural=True)
        return _EmptyImageList()

    def adjust_color_balance(self, factor: float) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        _check_adjust_color_balance_errors_and_warnings(factor, 0, plural=True)  # Disable channel check with channel=0
        return _EmptyImageList()

    def blur(self, radius: int) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        _check_blur_errors_and_warnings(radius, 1, plural=True)
        return _EmptyImageList()

    def sharpen(self, factor: float) -> ImageList:
        _EmptyImageList._warn_empty_image_list()
        _check_sharpen_errors_and_warnings(factor, plural=True)
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
