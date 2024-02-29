from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torchvision.transforms.v2 import functional as func2

from safeds.data.image.containers import Image
from safeds.data.image.containers._image_set import ImageSet

if TYPE_CHECKING:
    from safeds.data.image.containers._various_sized_image_set import _VariousSizedImageSet


class _FixedSizedImageSet(ImageSet):

    def __init__(self):
        self._tensor = None
        self._indices: list[int] = []

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
        image_set._indices = [index for image, index in images_ready_to_concat]

        return image_set

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImageSet):
            return NotImplemented
        if not isinstance(other, _FixedSizedImageSet):
            return False
        return (
            self._tensor.size() == other._tensor.size()
            and set(self.indices) == set(self.indices)
            and torch.all(torch.eq(self._tensor[torch.tensor(self.indices).argsort()], other._tensor[torch.tensor(other.indices).argsort()])).item()
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

    @property
    def indices(self) -> list[int]:
        return self._indices

    def get_image(self, index: int) -> Image:
        if index not in self.indices:
            raise KeyError(f'No image with index {index}')
        return Image(self._tensor[self.indices.index(index)])

    def index(self, image: Image) -> list[int]:
        if image not in self:
            return []
        return [self.indices[i] for i in (image._image_tensor == self._tensor).all(dim=[1, 2, 3]).nonzero()]

    def has_image(self, image: Image) -> bool:
        return not (image.width != self.widths[0] or image.height != self.heights[0]) and image._image_tensor in self._tensor

    def to_jpeg_files(self, path: str | Path | list[str] | list[Path]) -> None:
        pass

    def to_png_files(self, path: str | Path | list[str] | list[Path]) -> None:
        pass

    def to_images(self, indices: list[int] | None = None) -> list[Image]:
        if indices is None:
            indices = self.indices
        image_list = []
        for i in range(self._tensor.size(dim=0)):
            image_list.append(Image(self._tensor[i]))
        return [image_list[indices.index(i)] for i in sorted(indices)]

    def change_channel(self, channel: int) -> ImageSet:
        image_set = _FixedSizedImageSet()
        image_set._indices = self.indices
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

        if self._tensor.size(dim=2) == image_tensor.size(dim=1) and self._tensor.size(dim=3) == image_tensor.size(dim=2):
            image_set = _FixedSizedImageSet()
            image_set._indices = self.indices
            if image_tensor.size(dim=0) > self.channel:
                image_set = self.change_channel(image_tensor.size(dim=0))
            else:
                image_set._tensor = self._tensor
            if image_tensor.size(dim=0) < image_set._tensor.size(dim=1):
                if image_set._tensor.size(dim=1) == 3:
                    image_set._tensor = torch.cat([image_set._tensor, torch.cat([image_tensor, image_tensor, image_tensor], dim=0).unsqueeze(dim=0)])
                if image_tensor.size(dim=0) == 1:
                    image_set._tensor = torch.cat([image_set._tensor, torch.cat([image_tensor, image_tensor, image_tensor, torch.full(image_tensor.size(), 255)], dim=0).unsqueeze(dim=0)])
                else:
                    image_set._tensor = torch.cat([image_set._tensor, torch.cat([image_tensor, torch.full(image_tensor[0:1].size(), 255)], dim=0).unsqueeze(dim=0)])
            else:
                image_set._tensor = torch.cat([image_set._tensor, image_tensor.unsqueeze(dim=0)])
            image_set._indices.append(index)
        else:
            image_set = _VariousSizedImageSet._create_image_set([im._image_tensor for im in self.to_images()], self.indices)
            image_set._add_image_tensor(image_tensor, index)
        return image_set

    def remove_image_by_index(self, index: int | list[int]) -> ImageSet:
        if isinstance(index, int):
            index = [index]
        image_set = _FixedSizedImageSet()
        image_set._indices = [i - len([k for k in index if k < i]) for i in self.indices if i not in index]
        image_set._tensor = self._tensor[[i for i, v in enumerate(self.indices) if v not in index]]
        return image_set

    def remove_images_with_size(self, width: int, height: int) -> ImageSet:
        if self.widths[0] == width and self.heights[0] == height:
            return _FixedSizedImageSet()
        else:
            image_set = _FixedSizedImageSet()
            image_set._indices = self.indices
            image_set._tensor = self._tensor
            return image_set

    def remove_duplicate_images(self) -> ImageSet:
        image_set = _FixedSizedImageSet()
        image_set._tensor, new_indices = self._tensor.unique(dim=0, return_inverse=True)
        offset = 0
        indices = []
        for index_n, index_v in enumerate(new_indices):
            if list(new_indices).index(index_v) == index_n:
                indices.append(self.indices[index_v + offset] - offset)
            else:
                offset += 1
        image_set._indices = indices
        return image_set

    def shuffle_images(self) -> ImageSet:
        pass

    def resize(self, new_width: int, new_height: int) -> ImageSet:
        pass

    def convert_to_grayscale(self) -> ImageSet:
        image_set = _FixedSizedImageSet()
        image_set._tensor = func2.rgb_to_grayscale(self._tensor[:, 0:3])
        image_set._indices = self._indices
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
