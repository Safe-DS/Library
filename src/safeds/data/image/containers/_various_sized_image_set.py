from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from safeds.data.image.containers import Image
from safeds.data.image.containers._image_set import ImageSet
from safeds.exceptions import IndexOutOfBoundsError, DuplicateIndexError

if TYPE_CHECKING:
    from safeds.data.image.containers import _FixedSizedImageSet


class _VariousSizedImageSet(ImageSet):

    def __init__(self):
        self._image_set_dict: dict[tuple[int, int], ImageSet] = {}  # {image_size: image_set}
        self._indices_to_image_size_dict: dict[int, tuple[int, int]] = {}  # {index: image_size}

    @staticmethod
    def _create_image_set(images: list[Tensor], indices: list[int]) -> ImageSet:
        from safeds.data.image.containers import _FixedSizedImageSet

        max_channel = 0

        image_tensor_dict = {}
        image_index_dict = {}

        image_set = _VariousSizedImageSet()
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

        for size in image_tensor_dict.keys():
            image_set._image_set_dict[size] = _FixedSizedImageSet._create_image_set(image_tensor_dict[size], image_index_dict[size])
            image_set._indices_to_image_size_dict.update(zip(image_set._image_set_dict[size]._as_fixed_sized_image_set()._indices_to_tensor_positions.keys(), [size] * len(image_set._image_set_dict[size])))

        if max_channel > 1:
            image_set = image_set.change_channel(max_channel)

        return image_set

    def clone(self) -> ImageSet:
        cloned_image_set = self._clone_without_image_dict()
        for image_set_size, image_set in self._image_set_dict.items():
            cloned_image_set._image_set_dict[image_set_size] = image_set.clone()
        return cloned_image_set

    def _clone_without_image_dict(self) -> _VariousSizedImageSet:
        cloned_image_set = _VariousSizedImageSet()
        cloned_image_set._indices_to_image_size_dict = copy.deepcopy(self._indices_to_image_size_dict)
        return cloned_image_set

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImageSet):
            return NotImplemented
        if not isinstance(other, _VariousSizedImageSet) or set(other._image_set_dict) != set(self._image_set_dict):
            return False
        for image_set_key, image_set_value in self._image_set_dict.items():
            if image_set_value != other._image_set_dict[image_set_key]:
                return False
        return True

    @property
    def number_of_images(self) -> int:
        length = 0
        for image_set in self._image_set_dict.values():
            length += len(image_set)
        return length

    @property
    def widths(self) -> list[int]:
        widths = {}
        for image_set in self._image_set_dict.values():
            indices = image_set._as_fixed_sized_image_set()._tensor_positions_to_indices
            for i, index in enumerate(indices):
                widths[index] = image_set.widths[i]
        return [widths[index] for index in sorted(widths)]

    @property
    def heights(self) -> list[int]:
        heights = {}
        for image_set in self._image_set_dict.values():
            indices = image_set._as_fixed_sized_image_set()._tensor_positions_to_indices
            for i, index in enumerate(indices):
                heights[index] = image_set.heights[i]
        return [heights[index] for index in sorted(heights)]

    @property
    def channel(self) -> int:
        for image_set in self._image_set_dict.values():
            return image_set.channel

    @property
    def number_of_sizes(self) -> int:
        return len(self._image_set_dict)

    def get_image(self, index: int) -> Image:
        if index not in self._indices_to_image_size_dict:
            raise IndexOutOfBoundsError(index)
        return self._image_set_dict[self._indices_to_image_size_dict[index]].get_image(index)

    def index(self, image: Image) -> list[int]:
        indices = []
        for image_set in self._image_set_dict.values():
            if image in image_set:
                indices += image_set.index(image)
        return indices

    def has_image(self, image: Image) -> bool:
        return (image.width, image.height) in self._image_set_dict and self._image_set_dict[(image.width, image.height)].has_image(image)

    def to_jpeg_files(self, path: str | Path | list[str] | list[Path]) -> None:
        pass

    def to_png_files(self, path: str | Path | list[str] | list[Path]) -> None:
        pass

    def to_images(self, indices: list[int] | None = None) -> list[Image]:
        if indices is None:
            indices = sorted(self._indices_to_image_size_dict)
        else:
            wrong_indices = []
            for index in indices:
                if index not in self._indices_to_image_size_dict:
                    wrong_indices.append(index)
            if len(wrong_indices) == 0:
                raise IndexOutOfBoundsError(wrong_indices)
        images = []
        for index in indices:
            images.append(self._image_set_dict[self._indices_to_image_size_dict[index]].get_image(index))
        return images

    def change_channel(self, channel: int) -> ImageSet:
        image_set = self._clone_without_image_dict()
        for image_set_key, image_set_original in self._image_set_dict.items():
            image_set._image_set_dict[image_set_key] = image_set_original.change_channel(channel)
        return image_set

    def _add_image_tensor(self, image_tensor: Tensor, index: int) -> ImageSet:
        from safeds.data.image.containers import _FixedSizedImageSet

        if index in self._indices_to_image_size_dict:
            raise DuplicateIndexError(index)

        image_set = self.clone()._as_various_sized_image_set()
        size = (image_tensor.size(dim=2), image_tensor.size(dim=1))
        image_set._indices_to_image_size_dict[index] = size

        if size in self._image_set_dict:
            image_set._image_set_dict[size] = image_set._image_set_dict[size]._add_image_tensor(image_tensor, index)
        else:
            image_set._image_set_dict[size] = _FixedSizedImageSet._create_image_set([image_tensor], [index])

        if image_tensor.size(dim=0) != self.channel:
            image_set = image_set.change_channel(max(image_tensor.size(dim=0), self.channel))

        return image_set

    def add_images(self, images: list[Image] | ImageSet) -> ImageSet:
        from safeds.data.image.containers import _FixedSizedImageSet

        images_with_size = {}
        indices_for_images_with_size = {}
        current_index = max(self._indices_to_image_size_dict) + 1
        if isinstance(images, ImageSet):
            if images.number_of_sizes == 1:
                images_with_size[(images.widths[0], images.heights[0])] = images
                indices_for_images_with_size[(images.widths[0], images.heights[0])] = [index + current_index for index in images._as_fixed_sized_image_set()._tensor_positions_to_indices]
            else:
                for size, im_set in images._as_various_sized_image_set()._image_set_dict.items():
                    images_with_size[size] = im_set
                    indices_for_images_with_size[size] = [index + current_index for index in im_set._as_fixed_sized_image_set()._tensor_positions_to_indices]
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
        image_set = self.clone()._as_various_sized_image_set()
        smallest_channel = max_channel = self.channel
        for size, ims in images_with_size.items():
            new_indices = indices_for_images_with_size[size]
            if size in image_set._image_set_dict:
                if isinstance(ims, _FixedSizedImageSet):
                    ims_tensors = [im._image_tensor for im in ims.to_images()]
                else:
                    ims_tensors = [im._image_tensor for im in ims]
                image_set._image_set_dict[size] = _FixedSizedImageSet._create_image_set([im._image_tensor for im in image_set._image_set_dict[size].to_images()] + ims_tensors, image_set._image_set_dict[size]._as_fixed_sized_image_set()._tensor_positions_to_indices + new_indices)
            else:
                if isinstance(ims, _FixedSizedImageSet):
                    if smallest_channel > ims.channel:
                        smallest_channel = ims.channel
                    fixed_ims = ims.clone()._as_fixed_sized_image_set()
                    old_indices = list(fixed_ims._indices_to_tensor_positions.items())
                    fixed_ims._tensor_positions_to_indices = [new_indices[i] for i in sorted(range(len(new_indices)), key=sorted(range(len(new_indices)), key=old_indices.__getitem__).__getitem__)]
                    fixed_ims._calc_new_indices_to_tensor_positions()
                    image_set._image_set_dict[size] = fixed_ims
                else:
                    image_set._image_set_dict[size] = _FixedSizedImageSet._create_image_set([im._image_tensor for im in ims], new_indices)
                    if smallest_channel > image_set._image_set_dict[size].channel:
                        smallest_channel = image_set._image_set_dict[size].channel
            for i in new_indices:
                image_set._indices_to_image_size_dict[i] = size
            max_channel = max(max_channel,  image_set._image_set_dict[size].channel)
        if smallest_channel < max_channel:
            image_set = image_set.change_channel(max_channel)
        return image_set

    def remove_image_by_index(self, index: int | list[int]) -> ImageSet:
        if isinstance(index, int):
            index = [index]
        image_set = self._clone_without_image_dict()

        for image_set_key, image_set_original in self._image_set_dict.items():
            image_set._image_set_dict[image_set_key] = image_set_original.remove_image_by_index(index)
        [image_set._indices_to_image_size_dict.pop(i, None) for i in index]
        return image_set

    def remove_images_with_size(self, width: int, height: int) -> ImageSet:
        image_set = _VariousSizedImageSet()
        for image_set_key, image_set_original in self._image_set_dict.items():
            if (width, height) != image_set_key:
                image_set._image_set_dict[image_set_key] = image_set_original.clone()
        for index, size in self._indices_to_image_size_dict:
            if size != (width, height):
                image_set._indices_to_image_size_dict[index] = size
        return image_set

    def remove_duplicate_images(self) -> ImageSet:
        image_set = self._clone_without_image_dict()
        for image_set_key, image_set_original in self._image_set_dict.items():
            image_set._image_set_dict[image_set_key] = image_set_original.remove_duplicate_images()
        return image_set

    def shuffle_images(self) -> ImageSet:
        image_set = _VariousSizedImageSet()
        new_indices = list(self._indices_to_image_size_dict.keys())
        random.shuffle(new_indices)
        current_index = 0
        for image_set_key, image_set_original in self._image_set_dict.items():
            new_image_set = image_set_original.clone()._as_fixed_sized_image_set()
            new_image_set._tensor_positions_to_indices = new_indices[current_index:current_index + len(image_set_original)]
            new_image_set._indices_to_tensor_positions = new_image_set._calc_new_indices_to_tensor_positions()
            image_set._image_set_dict[image_set_key] = new_image_set
            for i in new_indices[current_index:current_index + len(image_set_original)]:
                image_set._indices_to_image_size_dict[i] = image_set_key
            current_index += len(image_set_original)
        return image_set

    def resize(self, new_width: int, new_height: int) -> _FixedSizedImageSet:
        from safeds.data.image.containers import _FixedSizedImageSet

        image_set_tensors = []
        image_set_indices = []
        for image_set_original in self._image_set_dict.values():
            image_set_new = image_set_original.resize(new_width, new_height)
            image_set_tensors.append(image_set_new._tensor)
            image_set_indices += image_set_new._tensor_positions_to_indices
        image_set = _FixedSizedImageSet()
        image_set._tensor = torch.cat(image_set_tensors, dim=0)
        image_set._tensor_positions_to_indices = image_set_indices
        image_set._indices_to_tensor_positions = image_set._calc_new_indices_to_tensor_positions()
        return image_set

    def convert_to_grayscale(self) -> ImageSet:
        image_set = self._clone_without_image_dict()
        for image_set_key, image_set_original in self._image_set_dict.items():
            image_set._image_set_dict[image_set_key] = image_set_original.convert_to_grayscale()
        return image_set

    def crop(self, x: int, y: int, width: int, height: int) -> _FixedSizedImageSet:
        from safeds.data.image.containers import _FixedSizedImageSet

        image_set_tensors = []
        image_set_indices = []
        for image_set_original in self._image_set_dict.values():
            image_set_new = image_set_original.crop(x, y, width, height)
            image_set_tensors.append(image_set_new._tensor)
            image_set_indices += image_set_new._tensor_positions_to_indices
        image_set = _FixedSizedImageSet()
        image_set._tensor = torch.cat(image_set_tensors, dim=0)
        image_set._tensor_positions_to_indices = image_set_indices
        image_set._indices_to_tensor_positions = image_set._calc_new_indices_to_tensor_positions()
        return image_set

    def flip_vertically(self) -> ImageSet:
        image_set = self._clone_without_image_dict()
        for image_set_key, image_set_original in self._image_set_dict.items():
            image_set._image_set_dict[image_set_key] = image_set_original.flip_vertically()
        return image_set

    def flip_horizontally(self) -> ImageSet:
        image_set = self._clone_without_image_dict()
        for image_set_key, image_set_original in self._image_set_dict.items():
            image_set._image_set_dict[image_set_key] = image_set_original.flip_horizontally()
        return image_set

    def adjust_brightness(self, factor: float) -> ImageSet:
        image_set = self._clone_without_image_dict()
        for image_set_key, image_set_original in self._image_set_dict.items():
            image_set._image_set_dict[image_set_key] = image_set_original.adjust_brightness(factor)
        return image_set

    def add_noise(self, standard_deviation: float) -> ImageSet:
        image_set = self._clone_without_image_dict()
        for image_set_key, image_set_original in self._image_set_dict.items():
            image_set._image_set_dict[image_set_key] = image_set_original.add_noise(standard_deviation)
        return image_set

    def adjust_contrast(self, factor: float) -> ImageSet:
        image_set = self._clone_without_image_dict()
        for image_set_key, image_set_original in self._image_set_dict.items():
            image_set._image_set_dict[image_set_key] = image_set_original.adjust_contrast(factor)
        return image_set

    def adjust_color_balance(self, factor: float) -> ImageSet:
        image_set = self._clone_without_image_dict()
        for image_set_key, image_set_original in self._image_set_dict.items():
            image_set._image_set_dict[image_set_key] = image_set_original.adjust_color_balance(factor)
        return image_set

    def blur(self, radius: int) -> ImageSet:
        image_set = self._clone_without_image_dict()
        for image_set_key, image_set_original in self._image_set_dict.items():
            image_set._image_set_dict[image_set_key] = image_set_original.blur(radius)
        return image_set

    def sharpen(self, factor: float) -> ImageSet:
        image_set = self._clone_without_image_dict()
        for image_set_key, image_set_original in self._image_set_dict.items():
            image_set._image_set_dict[image_set_key] = image_set_original.sharpen(factor)
        return image_set

    def invert_colors(self) -> ImageSet:
        image_set = self._clone_without_image_dict()
        for image_set_key, image_set_original in self._image_set_dict.items():
            image_set._image_set_dict[image_set_key] = image_set_original.invert_colors()
        return image_set

    def rotate_right(self) -> ImageSet:
        image_set = self._clone_without_image_dict()
        for image_set_key, image_set_original in self._image_set_dict.items():
            image_set._image_set_dict[image_set_key] = image_set_original.rotate_right()
        return image_set

    def rotate_left(self) -> ImageSet:
        image_set = self._clone_without_image_dict()
        for image_set_key, image_set_original in self._image_set_dict.items():
            image_set._image_set_dict[image_set_key] = image_set_original.rotate_left()
        return image_set

    def find_edges(self) -> ImageSet:
        image_set = self._clone_without_image_dict()
        for image_set_key, image_set_original in self._image_set_dict.items():
            image_set._image_set_dict[image_set_key] = image_set_original.find_edges()
        return image_set
