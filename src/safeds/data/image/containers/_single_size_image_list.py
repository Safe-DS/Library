from __future__ import annotations

import copy
import random
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as func2

from safeds.data.image.containers import Image, ImageList
from safeds.exceptions import DuplicateIndexError, IndexOutOfBoundsError, OutOfBoundsError, ClosedBound

if TYPE_CHECKING:
    from safeds.data.image.containers import _EmptyImageList, _MultiSizeImageList


class _SingleSizeImageList(ImageList):

    def __init__(self):
        self._tensor = None
        self._tensor_positions_to_indices: list[int] = []  # list[tensor_position] = index
        self._indices_to_tensor_positions: dict[int, int] = {}  # {index: tensor_position}

    @staticmethod
    def _create_image_list(images: list[Tensor], indices: list[int]) -> ImageList:
        from safeds.data.image.containers import _EmptyImageList

        if len(images) == 0:
            return _EmptyImageList()

        image_list = _SingleSizeImageList()
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
                image = torch.cat([image, image, image, torch.full(image.size(), 255, device=image.device)], dim=0)
            else:  # image channel 3 and max channel 4
                image = torch.cat([image, torch.full(image[0:1].size(), 255, device=image.device)], dim=0)
            images_ready_to_concat.append((image.unsqueeze(dim=0), index))
        image_list._tensor = torch.cat([image for image, index in images_ready_to_concat])
        image_list._tensor_positions_to_indices = [index for image, index in images_ready_to_concat]
        image_list._indices_to_tensor_positions = image_list._calc_new_indices_to_tensor_positions()

        return image_list

    def clone(self) -> ImageList:
        cloned_image_list = self._clone_without_tensor()
        cloned_image_list._tensor = self._tensor.detach().clone()
        return cloned_image_list

    def _clone_without_tensor(self) -> _SingleSizeImageList:
        cloned_image_list = _SingleSizeImageList()
        cloned_image_list._indices_to_tensor_positions = copy.deepcopy(self._indices_to_tensor_positions)
        cloned_image_list._tensor_positions_to_indices = copy.deepcopy(self._tensor_positions_to_indices)
        return cloned_image_list

    def _calc_new_indices_to_tensor_positions(self):
        _indices_to_tensor_positions: dict[int, int] = {}
        for i, index in enumerate(self._tensor_positions_to_indices):
            _indices_to_tensor_positions[index] = i
        return _indices_to_tensor_positions

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImageList):
            return NotImplemented
        if not isinstance(other, _SingleSizeImageList):
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
        return [self._tensor.size(dim=3)] * self.number_of_images

    @property
    def heights(self) -> list[int]:
        return [self._tensor.size(dim=2)] * self.number_of_images

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
        return not (image.width != self.widths[0] or image.height != self.heights[0] or image.channel != self.channel) and image._image_tensor in self._tensor

    def to_jpeg_files(self, path: str | Path | list[str] | list[Path]) -> None:
        pass

    def to_png_files(self, path: str | Path | list[str] | list[Path]) -> None:
        pass

    def to_images(self, indices: list[int] | None = None) -> list[Image]:
        if indices is None:
            indices = sorted(self._tensor_positions_to_indices)
        else:
            wrong_indices = []
            for index in indices:
                if index not in self._indices_to_tensor_positions:
                    wrong_indices.append(index)
            if len(wrong_indices) != 0:
                raise IndexOutOfBoundsError(wrong_indices)
        return [Image(self._tensor[self._indices_to_tensor_positions[index]]) for index in indices]

    def change_channel(self, channel: int) -> ImageList:
        image_list = self._clone_without_tensor()
        image_list._tensor = _SingleSizeImageList._change_channel_of_tensor(self._tensor, channel)
        return image_list

    @staticmethod
    def _change_channel_of_tensor(tensor: Tensor, channel: int) -> Tensor:
        if tensor.size(dim=-3) == channel:
            return tensor.detach().clone()
        elif tensor.size(dim=-3) == 1 and channel == 3:
            return torch.cat([tensor, tensor, tensor], dim=1)
        elif tensor.size(dim=-3) == 1 and channel == 4:
            return torch.cat([tensor, tensor, tensor, torch.full(tensor.size(), 255)], dim=1)
        elif tensor.size(dim=-3) in (3, 4) and channel == 1:
            return _SingleSizeImageList._convert_tensor_to_grayscale(tensor)[:, 0:1]
        elif tensor.size(dim=-3) == 3 and channel == 4:
            return torch.cat([tensor, torch.full(tensor[:, 0:1].size(), 255)], dim=1)
        elif tensor.size(dim=-3) == 4 and channel == 3:
            return tensor[:, 0:3].detach().clone()
        else:
            raise ValueError(f"Channel {channel} is not a valid channel option. Use either 1, 3 or 4")

    def _add_image_tensor(self, image_tensor: Tensor, index: int) -> ImageList:
        from safeds.data.image.containers import _MultiSizeImageList

        if index in self._indices_to_tensor_positions:
            raise DuplicateIndexError(index)

        if self._tensor.size(dim=2) == image_tensor.size(dim=1) and self._tensor.size(dim=3) == image_tensor.size(dim=2):
            image_list = self._clone_without_tensor()
            if image_tensor.size(dim=0) > self.channel:
                tensor = self.change_channel(image_tensor.size(dim=0))._as_single_size_image_list()._tensor
            else:
                tensor = self._tensor
            if image_tensor.size(dim=0) < tensor.size(dim=1):
                if tensor.size(dim=1) == 3:
                    image_list._tensor = torch.cat([tensor, torch.cat([image_tensor, image_tensor, image_tensor], dim=0).unsqueeze(dim=0)])
                elif image_tensor.size(dim=0) == 1:
                    image_list._tensor = torch.cat([tensor, torch.cat([image_tensor, image_tensor, image_tensor, torch.full(image_tensor.size(), 255)], dim=0).unsqueeze(dim=0)])
                else:
                    image_list._tensor = torch.cat([tensor, torch.cat([image_tensor, torch.full(image_tensor[0:1].size(), 255)], dim=0).unsqueeze(dim=0)])
            else:
                image_list._tensor = torch.cat([tensor, image_tensor.unsqueeze(dim=0)])
            image_list._tensor_positions_to_indices.append(index)
            image_list._indices_to_tensor_positions[index] = len(self)
        else:
            image_list = _MultiSizeImageList()
            image_list._image_list_dict[(self.widths[0], self.heights[0])] = self.clone()
            image_list._image_list_dict[(image_tensor.size(dim=2), image_tensor.size(dim=1))] = _SingleSizeImageList._create_image_list([image_tensor], [index])

            if image_tensor.size(dim=0) != self.channel:
                image_list = image_list.change_channel(max(image_tensor.size(dim=0), self.channel))
            for index in self._tensor_positions_to_indices:
                image_list._indices_to_image_size_dict[index] = (self.widths[0], self.heights[0])
            image_list._indices_to_image_size_dict[index] = (image_tensor.size(dim=2), image_tensor.size(dim=1))

        return image_list

    def add_images(self, images: list[Image] | ImageList) -> ImageList:
        from safeds.data.image.containers import _EmptyImageList, _MultiSizeImageList

        if isinstance(images, _EmptyImageList) or isinstance(images, list) and len(images) == 0:
            return self.clone()

        first_new_index = max(self._tensor_positions_to_indices) + 1
        if isinstance(images, list):
            images = ImageList.from_images(images)
        if isinstance(images, _MultiSizeImageList):
            image_list = _MultiSizeImageList()
            image_list._image_list_dict[(self.widths[0], self.heights[0])] = self.clone()
            for index in self._tensor_positions_to_indices:
                image_list._indices_to_image_size_dict[index] = (self.widths[0], self.heights[0])
            return image_list.add_images(images)
        if isinstance(images, _SingleSizeImageList):
            new_indices = [index + first_new_index for index in images._tensor_positions_to_indices]
            if images.widths[0] == self.widths[0] and images.heights[0] == self.heights[0]:
                image_list = self._clone_without_tensor()._as_single_size_image_list()
                image_list._tensor_positions_to_indices += new_indices
                if self.channel > images.channel:
                    image_list._tensor = torch.cat([self._tensor, _SingleSizeImageList._change_channel_of_tensor(images._tensor, self.channel)])
                elif self.channel < images.channel:
                    image_list._tensor = torch.cat([_SingleSizeImageList._change_channel_of_tensor(self._tensor, images.channel), images._tensor])
                else:
                    image_list._tensor = torch.cat([self._tensor, images._tensor])
                image_list._indices_to_tensor_positions = image_list._calc_new_indices_to_tensor_positions()
                return image_list
            else:
                image_list = _MultiSizeImageList()
                image_list._image_list_dict[(self.widths[0], self.heights[0])] = self.clone()
                image_list._image_list_dict[(images.widths[0], images.heights[0])] = images.clone()
                for index in self._tensor_positions_to_indices:
                    image_list._indices_to_image_size_dict[index] = (self.widths[0], self.heights[0])
                for index in images._tensor_positions_to_indices:
                    image_list._indices_to_image_size_dict[index + first_new_index] = (images.widths[0], images.heights[0])

                if images.channel != self.channel:
                    image_list = image_list.change_channel(max(images.channel, self.channel))
                return image_list

    def remove_image_by_index(self, index: int | list[int]) -> ImageList:
        from safeds.data.image.containers import _EmptyImageList

        if isinstance(index, int):
            index = [index]

        all_indices_to_remove = True
        for i in self._tensor_positions_to_indices:
            if i not in index:
                all_indices_to_remove = False
                continue
        if all_indices_to_remove:
            return _EmptyImageList()

        image_list = _SingleSizeImageList()
        image_list._tensor = self._tensor[[i for i, v in enumerate(self._tensor_positions_to_indices) if v not in index]].clone()
        image_list._tensor_positions_to_indices = [i - len([k for k in index if k < i]) for i in self._tensor_positions_to_indices if i not in index]
        image_list._indices_to_tensor_positions = image_list._calc_new_indices_to_tensor_positions()
        return image_list

    def remove_images_with_size(self, width: int, height: int) -> ImageList:
        from safeds.data.image.containers import _EmptyImageList

        if self.widths[0] == width and self.heights[0] == height:
            return _EmptyImageList()
        else:
            return self.clone()

    def remove_duplicate_images(self) -> ImageList:
        image_list = _SingleSizeImageList()
        tensor_cpu_unique, new_indices = self._tensor.cpu().unique(dim=0, return_inverse=True)  # Works somehow faster on cpu
        image_list._tensor = tensor_cpu_unique.to(self._tensor.device)
        indices, indices_to_remove = [], []
        offset_indices = []
        for index_n, index_v in enumerate(new_indices):
            if list(new_indices).index(index_v) == index_n:
                indices.append(self._tensor_positions_to_indices[index_v + len([offset for offset in offset_indices if index_v > offset])])
            else:
                offset_indices.append(index_v)
        indices_to_remove = [index_rem for index_rem in self._tensor_positions_to_indices if index_rem not in indices]
        for index_r in indices_to_remove:
            indices = [index - 1 if index > index_r else index for index in indices]
        image_list._tensor_positions_to_indices = indices
        image_list._indices_to_tensor_positions = image_list._calc_new_indices_to_tensor_positions()
        return image_list

    def shuffle_images(self) -> ImageList:
        image_list = self.clone()._as_single_size_image_list()
        random.shuffle(image_list._tensor_positions_to_indices)
        image_list._indices_to_tensor_positions = image_list._calc_new_indices_to_tensor_positions()
        return image_list

    def resize(self, new_width: int, new_height: int) -> ImageList:
        image_list = self._clone_without_tensor()
        image_list._tensor = func2.resize(self._tensor, size=[new_height, new_width], interpolation=InterpolationMode.NEAREST)
        return image_list

    def convert_to_grayscale(self) -> ImageList:
        image_list = self._clone_without_tensor()
        image_list._tensor = _SingleSizeImageList._convert_tensor_to_grayscale(self._tensor)
        return image_list

    @staticmethod
    def _convert_tensor_to_grayscale(tensor: Tensor) -> Tensor:
        if tensor.size(dim=-3) == 4:
            return torch.cat([func2.rgb_to_grayscale(tensor[:, 0:3], num_output_channels=3), tensor[:, 3].unsqueeze(dim=1)], dim=1)
        else:
            return func2.rgb_to_grayscale(tensor[:, 0:3], num_output_channels=3)

    def crop(self, x: int, y: int, width: int, height: int) -> ImageList:
        image_list = self._clone_without_tensor()
        image_list._tensor = func2.crop(self._tensor, x, y, height, width)
        return image_list

    def flip_vertically(self) -> ImageList:
        image_list = self._clone_without_tensor()
        image_list._tensor = func2.vertical_flip(self._tensor)
        return image_list

    def flip_horizontally(self) -> ImageList:
        image_list = self._clone_without_tensor()
        image_list._tensor = func2.horizontal_flip(self._tensor)
        return image_list

    def adjust_brightness(self, factor: float) -> ImageList:
        if factor < 0:
            raise OutOfBoundsError(factor, name="factor", lower_bound=ClosedBound(0))
        elif factor == 1:
            warnings.warn(
                "Brightness adjustment factor is 1.0, this will not make changes to the images.",
                UserWarning,
                stacklevel=2,
            )
        image_list = self._clone_without_tensor()
        if self.channel == 4:
            image_list._tensor = torch.cat([func2.adjust_brightness(self._tensor[:, 0:3], factor * 1.0), self._tensor[:, 3].unsqueeze(dim=1)], dim=1)
        else:
            image_list._tensor = func2.adjust_brightness(self._tensor, factor * 1.0)
        return image_list

    def add_noise(self, standard_deviation: float) -> ImageList:
        if standard_deviation < 0:
            raise OutOfBoundsError(standard_deviation, name="standard_deviation", lower_bound=ClosedBound(0))
        image_list = self._clone_without_tensor()
        image_list._tensor = self._tensor + torch.normal(0, standard_deviation, self._tensor.size()) * 255
        return image_list

    def adjust_contrast(self, factor: float) -> ImageList:
        if factor < 0:
            raise OutOfBoundsError(factor, name="factor", lower_bound=ClosedBound(0))
        elif factor == 1:
            warnings.warn(
                "Contrast adjustment factor is 1.0, this will not make changes to the images.",
                UserWarning,
                stacklevel=2,
            )
        image_list = self._clone_without_tensor()
        if self.channel == 4:
            image_list._tensor = torch.cat([func2.adjust_contrast(self._tensor[:, 0:3], factor * 1.0), self._tensor[:, 3].unsqueeze(dim=1)], dim=1)
        else:
            image_list._tensor = func2.adjust_contrast(self._tensor, factor * 1.0)
        return image_list

    def adjust_color_balance(self, factor: float) -> ImageList:
        if factor < 0:
            raise OutOfBoundsError(factor, name="factor", lower_bound=ClosedBound(0))
        elif factor == 1:
            warnings.warn(
                "Color adjustment factor is 1.0, this will not make changes to the images.",
                UserWarning,
                stacklevel=2,
            )
        elif self.channel == 1:
            warnings.warn(
                "Color adjustment will not have an affect on grayscale images with only one channel.",
                UserWarning,
                stacklevel=2,
            )
        image_list = self._clone_without_tensor()
        image_list._tensor = self.convert_to_grayscale()._as_single_size_image_list()._tensor * (1.0 - factor * 1.0) + self._tensor * (factor * 1.0)
        return image_list

    def blur(self, radius: int) -> ImageList:
        image_list = self._clone_without_tensor()
        image_list._tensor = func2.gaussian_blur(self._tensor, [radius * 2 + 1, radius * 2 + 1])
        return image_list

    def sharpen(self, factor: float) -> ImageList:
        if factor < 0:
            raise OutOfBoundsError(factor, name="factor", lower_bound=ClosedBound(0))
        elif factor == 1:
            warnings.warn(
                "Sharpen factor is 1.0, this will not make changes to the images.",
                UserWarning,
                stacklevel=2,
            )
        image_list = self._clone_without_tensor()
        if self.channel == 4:
            image_list._tensor = torch.cat([func2.adjust_sharpness(self._tensor[:, 0:3], factor * 1.0), self._tensor[:, 3].unsqueeze(dim=1)], dim=1)
        else:
            image_list._tensor = func2.adjust_sharpness(self._tensor, factor * 1.0)
        return image_list

    def invert_colors(self) -> ImageList:
        image_list = self._clone_without_tensor()
        if self.channel == 4:
            image_list._tensor = torch.cat([func2.invert(self._tensor[:, 0:3]), self._tensor[:, 3].unsqueeze(dim=1)], dim=1)
        else:
            image_list._tensor = func2.invert(self._tensor)
        return image_list

    def rotate_right(self) -> ImageList:
        image_list = self._clone_without_tensor()
        image_list._tensor = func2.rotate(self._tensor, -90, expand=True)
        return image_list

    def rotate_left(self) -> ImageList:
        image_list = self._clone_without_tensor()
        image_list._tensor = func2.rotate(self._tensor, 90, expand=True)
        return image_list

    def find_edges(self) -> ImageList:
        kernel = (
            Image._FILTER_EDGES_KERNEL.to("cpu")
        )
        edges_tensor = torch.clamp(
            torch.nn.functional.conv2d(
                self.convert_to_grayscale()._as_single_size_image_list()._tensor.float()[:, 0].unsqueeze(dim=1),
                kernel,
                padding="same",
            ),
            0,
            255,
        ).to(torch.uint8)
        image_list = self._clone_without_tensor()
        if self.channel == 3:
            image_list._tensor = edges_tensor.repeat(1, 3, 1, 1)
        elif self.channel == 4:
            image_list._tensor = torch.cat([edges_tensor.repeat(1, 3, 1, 1), self._tensor[:, 3].unsqueeze(dim=1)], dim=1)
        else:
            image_list._tensor = edges_tensor
        return image_list