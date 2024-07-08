from __future__ import annotations

import copy
import random
import sys
from typing import TYPE_CHECKING

from safeds._config import _init_default_device
from safeds._utils import _structural_hash
from safeds.data.image._utils._image_transformation_error_and_warning_checks import (
    _check_blur_errors_and_warnings,
    _check_crop_errors,
    _check_remove_images_with_size_errors,
    _check_resize_errors,
)
from safeds.data.image.containers import Image, ImageList
from safeds.exceptions import (
    DuplicateIndexError,
    IllegalFormatError,
    IndexOutOfBoundsError,
)

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

    from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
    from safeds.data.image.typing import ImageSize


class _MultiSizeImageList(ImageList):
    """
    An ImageList is a list of different images. It can hold different sizes of Images. The channel of all images is the same.

    This is the class for an ImageList with multiple different sizes.

    To create an `ImageList` call one of the following static methods:

    | Method                                                                        | Description                                              |
    | ----------------------------------------------------------------------------- | -------------------------------------------------------- |
    | [from_images][safeds.data.image.containers._image_list.ImageList.from_images] | Create an ImageList from a list of Images.               |
    | [from_files][safeds.data.image.containers._image_list.ImageList.from_files]   | Create an ImageList from a directory or a list of files. |
    """

    def __init__(self) -> None:
        self._image_list_dict: dict[tuple[int, int], ImageList] = {}  # {image_size: image_list}
        self._indices_to_image_size_dict: dict[int, tuple[int, int]] = {}  # {index: image_size}

    @staticmethod
    def _create_from_single_sized_image_lists(single_size_image_lists: list[_SingleSizeImageList]) -> ImageList:
        from safeds.data.image.containers._empty_image_list import _EmptyImageList

        if len(single_size_image_lists) == 0:
            return _EmptyImageList()
        elif len(single_size_image_lists) == 1:
            return single_size_image_lists[0]

        different_channels: bool = False
        max_channel: None | int = None

        image_list = _MultiSizeImageList()
        for single_size_image_list in single_size_image_lists:
            image_size = (single_size_image_list.widths[0], single_size_image_list.heights[0])
            image_list._image_list_dict[image_size] = single_size_image_list
            image_list._indices_to_image_size_dict.update(
                zip(
                    single_size_image_list._indices_to_tensor_positions.keys(),
                    [image_size] * len(single_size_image_list),
                    strict=False,
                ),
            )
            if max_channel is None:
                max_channel = single_size_image_list.channel
            elif max_channel < single_size_image_list.channel:
                different_channels = True
                max_channel = single_size_image_list.channel
            elif max_channel > single_size_image_list.channel:
                different_channels = True

        if different_channels:
            for size in image_list._image_list_dict:
                if max_channel is not None and image_list._image_list_dict[size].channel != max_channel:
                    image_list._image_list_dict[size] = image_list._image_list_dict[size].change_channel(
                        int(max_channel),
                    )
        return image_list

    @staticmethod
    def _create_image_list(images: list[Tensor], indices: list[int]) -> ImageList:
        from safeds.data.image.containers._empty_image_list import _EmptyImageList
        from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList

        if len(images) == 0:
            return _EmptyImageList()

        max_channel = 0

        image_tensor_dict = {}
        image_index_dict = {}

        image_list = _MultiSizeImageList()
        for index in indices:
            image = images.pop(0)
            size = (image.size(dim=2), image.size(dim=1))
            if size not in image_tensor_dict:
                image_tensor_dict[size] = [image]
                image_index_dict[size] = [index]
                max_channel = max(max_channel, image.size(dim=-3))
            else:
                image_tensor_dict[size].append(image)
                image_index_dict[size].append(index)
                max_channel = max(max_channel, image.size(dim=-3))

        for size in image_tensor_dict:
            image_list._image_list_dict[size] = _SingleSizeImageList._create_image_list(
                image_tensor_dict[size],
                image_index_dict[size],
            )
            image_list._indices_to_image_size_dict.update(
                zip(
                    image_list._image_list_dict[size]._as_single_size_image_list()._indices_to_tensor_positions.keys(),
                    [size] * len(image_list._image_list_dict[size]),
                    strict=False,
                ),
            )

        if max_channel > 1:
            image_list = image_list.change_channel(max_channel)._as_multi_size_image_list()

        return image_list

    def _clone(self) -> ImageList:
        cloned_image_list = self._clone_without_image_dict()
        for image_list_size, image_list in self._image_list_dict.items():
            cloned_image_list._image_list_dict[image_list_size] = image_list._clone()
        return cloned_image_list

    def _clone_without_image_dict(self) -> _MultiSizeImageList:
        """
        Clone this MultiSizeImageList to a new instance without the image data.

        Returns
        -------
        image_list:
            the cloned image list
        """
        cloned_image_list = _MultiSizeImageList()
        cloned_image_list._indices_to_image_size_dict = copy.deepcopy(self._indices_to_image_size_dict)
        return cloned_image_list

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImageList):
            return NotImplemented
        if not isinstance(other, _MultiSizeImageList) or set(other._image_list_dict) != set(self._image_list_dict):
            return False
        if self is other:
            return True
        for image_list_key, image_list_value in self._image_list_dict.items():
            if image_list_value != other._image_list_dict[image_list_key]:
                return False
        return True

    def __hash__(self) -> int:
        return _structural_hash([self._image_list_dict[image_size] for image_size in sorted(self._image_list_dict)])

    def __sizeof__(self) -> int:
        return (
            sys.getsizeof(self._image_list_dict)
            + sum(map(sys.getsizeof, self._image_list_dict.keys()))
            + sum(map(sys.getsizeof, self._image_list_dict.values()))
            + sys.getsizeof(self._indices_to_image_size_dict)
            + sum(map(sys.getsizeof, self._indices_to_image_size_dict.keys()))
            + sum(map(sys.getsizeof, self._indices_to_image_size_dict.values()))
        )

    @property
    def image_count(self) -> int:
        length = 0
        for image_list in self._image_list_dict.values():
            length += len(image_list)
        return length

    @property
    def widths(self) -> list[int]:
        widths = {}
        for image_list in self._image_list_dict.values():
            indices = image_list._as_single_size_image_list()._tensor_positions_to_indices
            for i, index in enumerate(indices):
                widths[index] = image_list.widths[i]
        return [widths[index] for index in sorted(widths)]

    @property
    def heights(self) -> list[int]:
        heights = {}
        for image_list in self._image_list_dict.values():
            indices = image_list._as_single_size_image_list()._tensor_positions_to_indices
            for i, index in enumerate(indices):
                heights[index] = image_list.heights[i]
        return [heights[index] for index in sorted(heights)]

    @property
    def channel(self) -> int:
        return next(iter(self._image_list_dict.values())).channel

    @property
    def sizes(self) -> list[ImageSize]:
        sizes = {}
        for image_list in self._image_list_dict.values():
            indices = image_list._as_single_size_image_list()._tensor_positions_to_indices
            for i, index in enumerate(indices):
                sizes[index] = image_list.sizes[i]
        return [sizes[index] for index in sorted(sizes)]

    @property
    def size_count(self) -> int:
        return len(self._image_list_dict)

    def get_image(self, index: int) -> Image:
        if index not in self._indices_to_image_size_dict:
            raise IndexOutOfBoundsError(index)
        return self._image_list_dict[self._indices_to_image_size_dict[index]].get_image(index)

    def index(self, image: Image) -> list[int]:
        indices = []
        for image_list in self._image_list_dict.values():
            if image in image_list:
                indices += image_list.index(image)
        return indices

    def has_image(self, image: Image) -> bool:
        return (image.width, image.height) in self._image_list_dict and self._image_list_dict[
            (image.width, image.height)
        ].has_image(image)

    def to_jpeg_files(self, path: str | Path | list[str | Path]) -> None:
        if self.channel == 4:
            raise IllegalFormatError("png")
        if isinstance(path, list):
            if len(path) == self.image_count:
                for image_size, image_list in self._image_list_dict.items():
                    image_list.to_jpeg_files(
                        [p for i, p in enumerate(path) if self._indices_to_image_size_dict[i] == image_size],
                    )
            elif len(path) == self.size_count:
                image_list_path: str | Path
                for image_list_path, image_list in zip(path, self._image_list_dict.values(), strict=False):
                    image_list.to_jpeg_files(image_list_path)
            else:
                raise ValueError(
                    "The path specified is invalid. Please provide either the path to a directory, a list of paths with one path for each image, or a list of paths with one path per image size.",
                )
        else:
            for image_list in self._image_list_dict.values():
                image_list.to_jpeg_files(path)

    def to_png_files(self, path: str | Path | list[str | Path]) -> None:
        if isinstance(path, list):
            if len(path) == self.image_count:
                for image_size, image_list in self._image_list_dict.items():
                    image_list.to_png_files(
                        [p for i, p in enumerate(path) if self._indices_to_image_size_dict[i] == image_size],
                    )
            elif len(path) == self.size_count:
                image_list_path: str | Path
                for image_list_path, image_list in zip(path, self._image_list_dict.values(), strict=False):
                    image_list.to_png_files(image_list_path)
            else:
                raise ValueError(
                    "The path specified is invalid. Please provide either the path to a directory, a list of paths with one path for each image, or a list of paths with one path per image size.",
                )
        else:
            for image_list in self._image_list_dict.values():
                image_list.to_png_files(path)

    def to_images(self, indices: list[int] | None = None) -> list[Image]:
        if indices is None:
            indices = sorted(self._indices_to_image_size_dict)
        else:
            wrong_indices = []
            for index in indices:
                if index not in self._indices_to_image_size_dict:
                    wrong_indices.append(index)
            if len(wrong_indices) != 0:
                raise IndexOutOfBoundsError(wrong_indices)
        images = []
        for index in indices:
            images.append(self._image_list_dict[self._indices_to_image_size_dict[index]].get_image(index))
        return images

    def change_channel(self, channel: int) -> ImageList:
        image_list = self._clone_without_image_dict()
        for image_list_key, image_list_original in self._image_list_dict.items():
            image_list._image_list_dict[image_list_key] = image_list_original.change_channel(channel)
        return image_list

    def _add_image_tensor(self, image_tensor: Tensor, index: int) -> ImageList:
        from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList

        if index in self._indices_to_image_size_dict:
            raise DuplicateIndexError(index)

        image_list = self._clone()._as_multi_size_image_list()
        size = (image_tensor.size(dim=2), image_tensor.size(dim=1))
        image_list._indices_to_image_size_dict[index] = size

        if size in self._image_list_dict:
            image_list._image_list_dict[size] = image_list._image_list_dict[size]._add_image_tensor(image_tensor, index)
        else:
            image_list._image_list_dict[size] = _SingleSizeImageList._create_image_list([image_tensor], [index])

        if image_tensor.size(dim=0) != self.channel:
            image_list = image_list.change_channel(
                max(image_tensor.size(dim=0), self.channel),
            )._as_multi_size_image_list()

        return image_list

    def add_images(self, images: list[Image] | ImageList) -> ImageList:
        from safeds.data.image.containers._empty_image_list import _EmptyImageList
        from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList

        if isinstance(images, _EmptyImageList) or isinstance(images, list) and len(images) == 0:
            return self

        indices_for_images_with_size = {}
        current_index = max(self._indices_to_image_size_dict) + 1
        image_list_with_size: dict[tuple[int, int], _SingleSizeImageList] = {}
        images_with_size: dict[tuple[int, int], list[Image]] = {}
        if isinstance(images, ImageList):
            if images.size_count == 1:
                image_list_with_size[(images.widths[0], images.heights[0])] = images._as_single_size_image_list()
                indices_for_images_with_size[(images.widths[0], images.heights[0])] = [
                    index + current_index for index in images._as_single_size_image_list()._tensor_positions_to_indices
                ]
            else:
                for size, im_list in images._as_multi_size_image_list()._image_list_dict.items():
                    image_list_with_size[size] = im_list._as_single_size_image_list()
                    indices_for_images_with_size[size] = [
                        index + current_index
                        for index in im_list._as_single_size_image_list()._tensor_positions_to_indices
                    ]
        else:
            for image in images:
                size = (image.width, image.height)
                if size in images_with_size:
                    images_with_size[size].append(image)
                    indices_for_images_with_size[size].append(current_index)
                else:
                    images_with_size[size] = [image]
                    indices_for_images_with_size[size] = [current_index]
                current_index += 1
        image_list = self._clone()._as_multi_size_image_list()
        smallest_channel = max_channel = self.channel
        for size, ims in (images_with_size | image_list_with_size).items():
            new_indices = indices_for_images_with_size[size]
            if size in image_list._image_list_dict:
                if isinstance(ims, _SingleSizeImageList):
                    ims_tensors = [im._image_tensor for im in ims.to_images()]
                else:
                    ims_tensors = [im._image_tensor for im in ims]
                image_list._image_list_dict[size] = _SingleSizeImageList._create_image_list(
                    [im._image_tensor for im in image_list._image_list_dict[size].to_images()] + ims_tensors,
                    image_list._image_list_dict[size]._as_single_size_image_list()._tensor_positions_to_indices
                    + new_indices,
                )
            elif isinstance(ims, _SingleSizeImageList):
                if smallest_channel > ims.channel:
                    smallest_channel = ims.channel
                fixed_ims = ims
                old_indices = list(fixed_ims._indices_to_tensor_positions.items())
                fixed_ims._tensor_positions_to_indices = [
                    new_indices[i]
                    for i in sorted(
                        range(len(new_indices)),
                        key=sorted(range(len(new_indices)), key=old_indices.__getitem__).__getitem__,
                    )
                ]
                fixed_ims._indices_to_tensor_positions = fixed_ims._calc_new_indices_to_tensor_positions()
                image_list._image_list_dict[size] = fixed_ims
            else:
                image_list._image_list_dict[size] = _SingleSizeImageList._create_image_list(
                    [im._image_tensor for im in ims],
                    new_indices,
                )
                if smallest_channel > image_list._image_list_dict[size].channel:
                    smallest_channel = image_list._image_list_dict[size].channel
            for i in new_indices:
                image_list._indices_to_image_size_dict[i] = size
            max_channel = max(max_channel, image_list._image_list_dict[size].channel)
        if smallest_channel < max_channel:
            image_list = image_list.change_channel(max_channel)._as_multi_size_image_list()
        return image_list

    def remove_image_by_index(self, index: int | list[int]) -> ImageList:
        if isinstance(index, int):
            index = [index]

        invalid_indices = []
        for _i in index:
            if _i not in self._indices_to_image_size_dict:
                invalid_indices.append(_i)
        if len(invalid_indices) > 0:
            raise IndexOutOfBoundsError(invalid_indices)

        return self._remove_image_by_index_ignore_invalid(index)

    def _remove_image_by_index_ignore_invalid(self, index: int | list[int]) -> ImageList:
        from safeds.data.image.containers._empty_image_list import _EmptyImageList
        from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList

        if isinstance(index, int):
            index = [index]
        image_list = self._clone_without_image_dict()

        for image_list_key, image_list_original in self._image_list_dict.items():
            new_single_size_image_list = (
                image_list_original._as_single_size_image_list()._remove_image_by_index_ignore_invalid(index)
            )
            if isinstance(new_single_size_image_list, _SingleSizeImageList):
                image_list._image_list_dict[image_list_key] = new_single_size_image_list
        [image_list._indices_to_image_size_dict.pop(i, None) for i in index]

        if len(image_list._image_list_dict) == 0:
            return _EmptyImageList()
        elif len(image_list._image_list_dict) == 1:
            return next(iter(image_list._image_list_dict.values()))
        else:
            return image_list

    def remove_images_with_size(self, width: int, height: int) -> ImageList:
        import torch

        _init_default_device()

        _check_remove_images_with_size_errors(width, height)
        if (width, height) not in self._image_list_dict:
            return self
        if len(self._image_list_dict) == 2:
            single_size_image_list = (
                self._image_list_dict[
                    next(iter([key for key in list(self._image_list_dict.keys()) if key != (width, height)]))
                ]
                ._clone()
                ._as_single_size_image_list()
            )
            single_size_image_list._tensor_positions_to_indices = torch.sort(
                torch.Tensor(single_size_image_list._tensor_positions_to_indices),
            )[1].tolist()
            single_size_image_list._indices_to_tensor_positions = (
                single_size_image_list._calc_new_indices_to_tensor_positions()
            )
            return single_size_image_list

        image_list = _MultiSizeImageList()
        for image_list_key, image_list_original in self._image_list_dict.items():
            if (width, height) != image_list_key:
                image_list._image_list_dict[image_list_key] = image_list_original
        for index, size in self._indices_to_image_size_dict.items():
            if size != (width, height):
                image_list._indices_to_image_size_dict[index] = size
        return image_list

    def remove_duplicate_images(self) -> ImageList:
        image_list = _MultiSizeImageList()
        indices_to_remove = []
        for image_list_key, image_list_original in self._image_list_dict.items():
            image_list_duplicates_removed = image_list_original.remove_duplicate_images()
            indices_to_remove += [
                index
                for index in image_list_original._as_single_size_image_list()._tensor_positions_to_indices
                if index not in image_list_duplicates_removed._as_single_size_image_list()._tensor_positions_to_indices
            ]
            image_list._image_list_dict[image_list_key] = image_list_duplicates_removed
        for index_rem in indices_to_remove:
            for image_list_duplicates_removed in image_list._image_list_dict.values():
                image_list_duplicates_removed._as_single_size_image_list()._tensor_positions_to_indices = [
                    index - 1 if index > index_rem else index
                    for index in image_list_duplicates_removed._as_single_size_image_list()._tensor_positions_to_indices
                ]
                image_list_duplicates_removed._as_single_size_image_list()._indices_to_tensor_positions = (
                    image_list_duplicates_removed._as_single_size_image_list()._calc_new_indices_to_tensor_positions()
                )
        next_index = 0
        for index_key, index_value in sorted(self._indices_to_image_size_dict.items()):
            if index_key not in indices_to_remove:
                image_list._indices_to_image_size_dict[next_index] = index_value
                next_index += 1
        return image_list

    def shuffle_images(self) -> ImageList:
        image_list = _MultiSizeImageList()
        new_indices = list(self._indices_to_image_size_dict.keys())
        random.shuffle(new_indices)
        current_index = 0
        for image_list_key, image_list_original in self._image_list_dict.items():
            new_image_list = image_list_original._clone()._as_single_size_image_list()
            new_image_list._tensor_positions_to_indices = new_indices[
                current_index : current_index + len(image_list_original)
            ]
            new_image_list._indices_to_tensor_positions = new_image_list._calc_new_indices_to_tensor_positions()
            image_list._image_list_dict[image_list_key] = new_image_list
            for i in new_indices[current_index : current_index + len(image_list_original)]:
                image_list._indices_to_image_size_dict[i] = image_list_key
            current_index += len(image_list_original)
        return image_list

    def resize(self, new_width: int, new_height: int) -> ImageList:
        import torch

        _init_default_device()

        from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList

        _check_resize_errors(new_width, new_height)

        image_list_indices = []
        image_list = _SingleSizeImageList()
        image_list._tensor = torch.empty(len(self), self.channel, new_height, new_width, dtype=torch.uint8)
        current_start_index = 0
        for image_list_original in self._image_list_dict.values():
            image_list_new = image_list_original.resize(new_width, new_height)._as_single_size_image_list()
            end = current_start_index + len(image_list_original)
            image_list._tensor[current_start_index:end] = image_list_new._tensor
            image_list_indices += image_list_new._tensor_positions_to_indices
            current_start_index = end
        image_list._tensor_positions_to_indices = image_list_indices
        image_list._indices_to_tensor_positions = image_list._calc_new_indices_to_tensor_positions()
        return image_list

    def convert_to_grayscale(self) -> ImageList:
        image_list = self._clone_without_image_dict()
        for image_list_key, image_list_original in self._image_list_dict.items():
            image_list._image_list_dict[image_list_key] = image_list_original.convert_to_grayscale()
        return image_list

    def crop(self, x: int, y: int, width: int, height: int) -> ImageList:
        import torch

        _init_default_device()

        from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList

        _check_crop_errors(x, y, width, height)

        image_list_indices = []
        image_list = _SingleSizeImageList()
        image_list._tensor = torch.empty(len(self), self.channel, width, height, dtype=torch.uint8)
        current_start_index = 0
        for image_list_original in self._image_list_dict.values():
            image_list_new = image_list_original.crop(x, y, width, height)._as_single_size_image_list()
            end = current_start_index + len(image_list_original)
            image_list._tensor[current_start_index:end] = image_list_new._tensor
            image_list_indices += image_list_new._tensor_positions_to_indices
            current_start_index = end
        image_list._tensor_positions_to_indices = image_list_indices
        image_list._indices_to_tensor_positions = image_list._calc_new_indices_to_tensor_positions()
        return image_list

    def flip_vertically(self) -> ImageList:
        image_list = self._clone_without_image_dict()
        for image_list_key, image_list_original in self._image_list_dict.items():
            image_list._image_list_dict[image_list_key] = image_list_original.flip_vertically()
        return image_list

    def flip_horizontally(self) -> ImageList:
        image_list = self._clone_without_image_dict()
        for image_list_key, image_list_original in self._image_list_dict.items():
            image_list._image_list_dict[image_list_key] = image_list_original.flip_horizontally()
        return image_list

    def adjust_brightness(self, factor: float) -> ImageList:
        image_list = self._clone_without_image_dict()
        for image_list_key, image_list_original in self._image_list_dict.items():
            image_list._image_list_dict[image_list_key] = image_list_original.adjust_brightness(factor)
        return image_list

    def add_noise(self, standard_deviation: float) -> ImageList:
        image_list = self._clone_without_image_dict()
        for image_list_key, image_list_original in self._image_list_dict.items():
            image_list._image_list_dict[image_list_key] = image_list_original.add_noise(standard_deviation)
        return image_list

    def adjust_contrast(self, factor: float) -> ImageList:
        image_list = self._clone_without_image_dict()
        for image_list_key, image_list_original in self._image_list_dict.items():
            image_list._image_list_dict[image_list_key] = image_list_original.adjust_contrast(factor)
        return image_list

    def adjust_color_balance(self, factor: float) -> ImageList:
        image_list = self._clone_without_image_dict()
        for image_list_key, image_list_original in self._image_list_dict.items():
            image_list._image_list_dict[image_list_key] = image_list_original.adjust_color_balance(factor)
        return image_list

    def blur(self, radius: int) -> ImageList:
        _check_blur_errors_and_warnings(radius, min(*self.widths, *self.heights), plural=True)
        image_list = self._clone_without_image_dict()
        for image_list_key, image_list_original in self._image_list_dict.items():
            image_list._image_list_dict[image_list_key] = image_list_original.blur(radius)
        return image_list

    def sharpen(self, factor: float) -> ImageList:
        image_list = self._clone_without_image_dict()
        for image_list_key, image_list_original in self._image_list_dict.items():
            image_list._image_list_dict[image_list_key] = image_list_original.sharpen(factor)
        return image_list

    def invert_colors(self) -> ImageList:
        image_list = self._clone_without_image_dict()
        for image_list_key, image_list_original in self._image_list_dict.items():
            image_list._image_list_dict[image_list_key] = image_list_original.invert_colors()
        return image_list

    def rotate_right(self) -> ImageList:
        image_list = self._clone_without_image_dict()
        for image_list_key, image_list_original in self._image_list_dict.items():
            image_list._image_list_dict[image_list_key] = image_list_original.rotate_right()
        return image_list

    def rotate_left(self) -> ImageList:
        image_list = self._clone_without_image_dict()
        for image_list_key, image_list_original in self._image_list_dict.items():
            image_list._image_list_dict[image_list_key] = image_list_original.rotate_left()
        return image_list

    def find_edges(self) -> ImageList:
        image_list = self._clone_without_image_dict()
        for image_list_key, image_list_original in self._image_list_dict.items():
            image_list._image_list_dict[image_list_key] = image_list_original.find_edges()
        return image_list
