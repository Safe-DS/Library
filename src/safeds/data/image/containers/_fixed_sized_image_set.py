from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torchvision.transforms.v2 import functional as func2

from safeds.data.image.containers import Image
from safeds.data.image.containers._image_set import ImageSet
from safeds.exceptions import DuplicateIndexError, IndexOutOfBoundsError

if TYPE_CHECKING:
    from safeds.data.image.containers._various_sized_image_set import _VariousSizedImageSet


class _FixedSizedImageSet(ImageSet):

    def __init__(self):
        self._tensor = None
        self._tensor_positions_to_indices: list[int] = []  # list[tensor_position] = index
        self._indices_to_tensor_positions: dict[int, int] = {}  # {index: tensor_position}

    @staticmethod
    def _create_image_set(images: list[Tensor], indices: list[int]) -> ImageSet:
        image_set = _FixedSizedImageSet()
        images_ready_to_concat = []
        images_with_correct_channel = []
        images_with_less_channels = []
        max_channel = 0
        for image, index in zip(images, indices):
            if max_channel < image.size(dim=-3):  # all images have to increase their channel
                images_with_less_channels += images_with_correct_channel
                images_with_correct_channel = [(image, index)]
                max_channel = image.size(dim=-3)
            elif max_channel > image.size(dim=-3):  # current image has to increase its channel
                images_with_less_channels.append((image, index))
            else:  # current image has same channel as max_channel
                images_with_correct_channel.append((image, index))
        for image, index in images_with_correct_channel:
            images_ready_to_concat.append((image.unsqueeze(dim=0), index))
        for image, index in images_with_less_channels:
            if max_channel == 3:  # image channel 1 and max channel 3
                image = torch.cat([image, image, image], dim=0)
            elif image.size(dim=0) == 1:  # image channel 1 and max channel 4
                image = torch.cat([image, image, image, torch.full(image.size(), 255)], dim=0)
            else:  # image channel 3 and max channel 4
                image = torch.cat([image, torch.full(image[0:1].size(), 255)], dim=0)
            images_ready_to_concat.append((image.unsqueeze(dim=0), index))
        image_set._tensor = torch.cat([image for image, index in images_ready_to_concat])
        image_set._tensor_positions_to_indices = [index for image, index in images_ready_to_concat]
        image_set._indices_to_tensor_positions = image_set._calc_new_indices_to_tensor_positions()

        return image_set

    def clone(self) -> ImageSet:
        cloned_image_set = self._clone_without_tensor()
        cloned_image_set._tensor = self._tensor.detach().clone()
        return cloned_image_set

    def _clone_without_tensor(self) -> _FixedSizedImageSet:
        cloned_image_set = _FixedSizedImageSet()
        cloned_image_set._indices_to_tensor_positions = copy.deepcopy(self._indices_to_tensor_positions)
        cloned_image_set._tensor_positions_to_indices = copy.deepcopy(self._tensor_positions_to_indices)
        return cloned_image_set

    def _calc_new_indices_to_tensor_positions(self):
        _indices_to_tensor_positions: dict[int, int] = {}
        for i, index in enumerate(self._tensor_positions_to_indices):
            _indices_to_tensor_positions[index] = i
        return _indices_to_tensor_positions

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImageSet):
            return NotImplemented
        if not isinstance(other, _FixedSizedImageSet):
            return False
        return (
            self._tensor.size() == other._tensor.size()
            and set(self._tensor_positions_to_indices) == set(self._tensor_positions_to_indices)
            and set(self._indices_to_tensor_positions) == set(self._indices_to_tensor_positions)
            and torch.all(torch.eq(self._tensor[torch.tensor(self._tensor_positions_to_indices).argsort()], other._tensor[torch.tensor(other._tensor_positions_to_indices).argsort()])).item()
        )

    @property
    def number_of_images(self) -> int:
        return self._tensor.size(dim=0)

    @property
    def widths(self) -> list[int]:
        return [self._tensor.size(dim=3)]

    @property
    def heights(self) -> list[int]:
        return [self._tensor.size(dim=2)]

    @property
    def channel(self) -> int:
        return self._tensor.size(dim=1)

    @property
    def number_of_sizes(self) -> int:
        return 1

    def get_image(self, index: int) -> Image:
        if index not in self._indices_to_tensor_positions:
            raise IndexOutOfBoundsError(index)
        return Image(self._tensor[self._indices_to_tensor_positions[index]])

    def index(self, image: Image) -> list[int]:
        if image not in self:
            return []
        return [self._tensor_positions_to_indices[i] for i in (image._image_tensor == self._tensor).all(dim=[1, 2, 3]).nonzero()]

    def has_image(self, image: Image) -> bool:
        return not (image.width != self.widths[0] or image.height != self.heights[0]) and image._image_tensor in self._tensor

    def to_jpeg_files(self, path: str | Path | list[str] | list[Path]) -> None:
        pass

    def to_png_files(self, path: str | Path | list[str] | list[Path]) -> None:
        pass

    def to_images(self, indices: list[int] | None = None) -> list[Image]:
        if indices is None:
            indices = self._tensor_positions_to_indices
        else:
            wrong_indices = []
            for index in indices:
                if index not in self._indices_to_tensor_positions:
                    wrong_indices.append(index)
            if len(wrong_indices) == 0:
                raise IndexOutOfBoundsError(wrong_indices)
        return [Image(self._tensor[self._indices_to_tensor_positions[index]]) for index in indices]

    def change_channel(self, channel: int) -> ImageSet:
        image_set = self._clone_without_tensor()
        if self.channel == channel:
            image_set._tensor = self._tensor
        elif self.channel == 1 and channel == 3:
            image_set._tensor = torch.cat([self._tensor, self._tensor, self._tensor], dim=1)
        elif self.channel == 1 and channel == 4:
            image_set._tensor = torch.cat([self._tensor, self._tensor, self._tensor, torch.full(self._tensor.size(), 255)], dim=1)
        elif self.channel in (3, 4) and channel == 1:
            image_set._tensor = self.convert_to_grayscale()._as_fixed_sized_image_set()._tensor[:, 0:1]
        elif self.channel == 3 and channel == 4:
            image_set._tensor = torch.cat([self._tensor, torch.full(self._tensor[:, 0:1].size(), 255)], dim=1)
        elif self.channel == 4 and channel == 3:
            image_set._tensor = self._tensor[:, 0:3]
        else:
            image_set._tensor = self._tensor
        return image_set

    def _add_image_tensor(self, image_tensor: Tensor, index: int) -> ImageSet:
        from safeds.data.image.containers import _VariousSizedImageSet

        if index in self._indices_to_tensor_positions:
            raise DuplicateIndexError(index)

        if self._tensor.size(dim=2) == image_tensor.size(dim=1) and self._tensor.size(dim=3) == image_tensor.size(dim=2):
            image_set = self._clone_without_tensor()
            if image_tensor.size(dim=0) > self.channel:
                tensor = self.change_channel(image_tensor.size(dim=0))._as_fixed_sized_image_set()._tensor
            else:
                tensor = self._tensor
            if image_tensor.size(dim=0) < tensor.size(dim=1):
                if tensor.size(dim=1) == 3:
                    image_set._tensor = torch.cat([tensor, torch.cat([image_tensor, image_tensor, image_tensor], dim=0).unsqueeze(dim=0)])
                if tensor.size(dim=0) == 1:
                    image_set._tensor = torch.cat([tensor, torch.cat([image_tensor, image_tensor, image_tensor, torch.full(image_tensor.size(), 255)], dim=0).unsqueeze(dim=0)])
                else:
                    image_set._tensor = torch.cat([tensor, torch.cat([image_tensor, torch.full(image_tensor[0:1].size(), 255)], dim=0).unsqueeze(dim=0)])
            else:
                image_set._tensor = torch.cat([tensor, image_tensor.unsqueeze(dim=0)])
            image_set._tensor_positions_to_indices.append(index)
            image_set._indices_to_tensor_positions[index] = len(self)
        else:
            image_set = _VariousSizedImageSet()
            image_set._image_set_dict[(self.widths[0], self.heights[0])] = self.clone()
            image_set._image_set_dict[(image_tensor.size(dim=2), image_tensor.size(dim=1))] = _FixedSizedImageSet._create_image_set([image_tensor], [index])

            if image_tensor.size(dim=0) != self.channel:
                image_set = image_set.change_channel(max(image_tensor.size(dim=0), self.channel))
            for index in self._tensor_positions_to_indices:
                image_set._indices_to_image_size_dict[index] = (self.widths[0], self.heights[0])
            image_set._indices_to_image_size_dict[index] = (image_tensor.size(dim=2), image_tensor.size(dim=1))

        return image_set

    def add_images(self, images: list[Image] | ImageSet) -> ImageSet:
        max_index = max(self._tensor_positions_to_indices)
        new_indices = list(range(max_index, max_index + 1 + len(images)))
        if isinstance(images, list):
            images = ImageSet.from_images(images, new_indices)
        if isinstance(images, _VariousSizedImageSet):
            image_set = _VariousSizedImageSet()
            image_set._image_set_dict[(self.widths[0], self.heights[0])] = self.clone()
            for index in self._tensor_positions_to_indices:
                image_set._indices_to_image_size_dict[index] = (self.widths[0], self.heights[0])
            return image_set.add_images(images)
        if isinstance(images, _FixedSizedImageSet):
            if images.widths == self.widths and images.heights == self.heights:
                return _FixedSizedImageSet._create_image_set([image._image_tensor for image in self.to_images() + images.to_images()], self._tensor_positions_to_indices + new_indices)
            else:
                image_set = _VariousSizedImageSet()
                image_set._image_set_dict[(self.widths[0], self.heights[0])] = self.clone()
                image_set._image_set_dict[(images.widths[0], images.heights[0])] = images.clone()
                for index in self._tensor_positions_to_indices:
                    image_set._indices_to_image_size_dict[index] = (self.widths[0], self.heights[0])
                for index in images._tensor_positions_to_indices:
                    image_set._indices_to_image_size_dict[index] = (images.widths[0], images.heights[0])

                if images.channel != self.channel:
                    image_set = image_set.change_channel(max(images.channel, self.channel))
                return image_set

    def remove_image_by_index(self, index: int | list[int]) -> ImageSet:
        if isinstance(index, int):
            index = [index]
        image_set = _FixedSizedImageSet()
        image_set._tensor = self._tensor[[i for i, v in enumerate(self._tensor_positions_to_indices) if v not in index]].clone()
        image_set._tensor_positions_to_indices = [i - len([k for k in index if k < i]) for i in self._tensor_positions_to_indices if i not in index]  # TODO: should the indices be changed?
        image_set._indices_to_tensor_positions = image_set._calc_new_indices_to_tensor_positions()
        return image_set

    def remove_images_with_size(self, width: int, height: int) -> ImageSet:
        if self.widths[0] == width and self.heights[0] == height:
            return _FixedSizedImageSet()
        else:
            return self.clone()

    def remove_duplicate_images(self) -> ImageSet:
        image_set = _FixedSizedImageSet()
        image_set._tensor, new_indices = self._tensor.unique(dim=0, return_inverse=True)
        offset = 0
        indices = []
        for index_n, index_v in enumerate(new_indices):
            if list(new_indices).index(index_v) == index_n:
                indices.append(self._tensor_positions_to_indices[index_v + offset] - offset)
            else:
                offset += 1
        image_set._tensor_positions_to_indices = indices
        image_set._indices_to_tensor_positions = image_set._calc_new_indices_to_tensor_positions()
        return image_set

    def shuffle_images(self) -> ImageSet:
        pass

    def resize(self, new_width: int, new_height: int) -> ImageSet:
        pass

    def convert_to_grayscale(self) -> ImageSet:
        image_set = self._clone_without_tensor()
        image_set._tensor = func2.rgb_to_grayscale(self._tensor[:, 0:3])
        return image_set

    def crop(self, x: int, y: int, width: int, height: int) -> ImageSet:
        pass

    def flip_vertically(self) -> ImageSet:
        pass

    def flip_horizontally(self) -> ImageSet:
        pass

    def adjust_brightness(self, factor: float) -> ImageSet:
        pass

    def add_noise(self, standard_deviation: float) -> ImageSet:
        pass

    def adjust_contrast(self, factor: float) -> ImageSet:
        pass

    def adjust_color_balance(self, factor: float) -> ImageSet:
        pass

    def blur(self, radius: int) -> ImageSet:
        pass

    def sharpen(self, factor: float) -> ImageSet:
        pass

    def invert_colors(self) -> ImageSet:
        pass

    def rotate_right(self) -> ImageSet:
        pass

    def rotate_left(self) -> ImageSet:
        pass

    def find_edges(self) -> ImageSet:
        pass
