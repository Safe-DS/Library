from __future__ import annotations

import copy
import math
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from safeds._config import _get_device, _init_default_device
from safeds._utils import _structural_hash
from safeds.data.image._utils._image_transformation_error_and_warning_checks import (
    _check_add_noise_errors,
    _check_adjust_brightness_errors_and_warnings,
    _check_adjust_color_balance_errors_and_warnings,
    _check_adjust_contrast_errors_and_warnings,
    _check_blur_errors_and_warnings,
    _check_crop_errors,
    _check_crop_warnings,
    _check_remove_images_with_size_errors,
    _check_resize_errors,
    _check_sharpen_errors_and_warnings,
)
from safeds.data.image.containers._image import Image
from safeds.data.image.containers._image_list import ImageList
from safeds.data.image.typing import ImageSize
from safeds.exceptions import (
    DuplicateIndexError,
    IllegalFormatError,
    IndexOutOfBoundsError,
)

if TYPE_CHECKING:
    from torch import Tensor

    from safeds.data.image.containers._multi_size_image_list import _MultiSizeImageList


class _SingleSizeImageList(ImageList):
    """
    An ImageList is a list of different images. It can hold different sizes of Images. The channel of all images is the same.

    This is the class for an ImageList with only one size for all images.

    To create an `ImageList` call one of the following static methods:

    | Method                                                                        | Description                                              |
    | ----------------------------------------------------------------------------- | -------------------------------------------------------- |
    | [from_images][safeds.data.image.containers._image_list.ImageList.from_images] | Create an ImageList from a list of Images.               |
    | [from_files][safeds.data.image.containers._image_list.ImageList.from_files]   | Create an ImageList from a directory or a list of files. |
    """

    def __init__(self) -> None:
        import torch

        _init_default_device()

        self._next_batch_index = 0
        self._batch_size = 1

        self._tensor: Tensor = torch.empty([])
        self._tensor_positions_to_indices: list[int] = []  # list[tensor_position] = index
        self._indices_to_tensor_positions: dict[int, int] = {}  # {index: tensor_position}

    @staticmethod
    def _create_image_list_from_files(
        images: dict[int, list[str]],
        image_count: int,
        max_channel: int,
        width: int,
        height: int,
        indices: dict[int, list[int]],
        max_files_per_thread_package: int,
    ) -> tuple[ImageList, list[ImageList._FromFileThreadPackage]]:
        import torch

        _init_default_device()

        from safeds.data.image.containers._empty_image_list import _EmptyImageList

        if len(images) == 0 or image_count == 0:
            return _EmptyImageList(), []

        image_list = _SingleSizeImageList()

        images_tensor = torch.empty(
            image_count,
            max_channel,
            height,
            width,
            dtype=torch.uint8,
            device=_get_device(),
        )

        thread_packages: list[ImageList._FromFileThreadPackage] = []
        current_thread_channel: int | None = None
        current_thread_channel_files: list[str] = []
        current_thread_start_index: int = 0
        while image_count - current_thread_start_index > 0:
            num_of_files = min(max_files_per_thread_package, image_count - current_thread_start_index)
            while num_of_files > 0:
                if current_thread_channel is None or len(current_thread_channel_files) == 0:
                    current_thread_channel = next(iter(images.keys()))
                    current_thread_channel_files = images.pop(current_thread_channel)
                next_package_size = min(num_of_files, len(current_thread_channel_files))
                package = ImageList._FromFileThreadPackage(
                    current_thread_channel_files[:next_package_size],
                    current_thread_channel,
                    max_channel,
                    width,
                    height,
                    images_tensor,
                    current_thread_start_index,
                )
                current_thread_start_index += next_package_size
                num_of_files -= next_package_size
                current_thread_channel_files = current_thread_channel_files[next_package_size:]
                thread_packages.append(package)

        image_list._tensor = images_tensor
        image_list._tensor_positions_to_indices = [i for j in indices.values() for i in j]
        image_list._indices_to_tensor_positions = image_list._calc_new_indices_to_tensor_positions()
        return image_list, thread_packages

    @staticmethod
    def _create_image_list(images: list[Tensor], indices: list[int]) -> ImageList:
        import torch

        _init_default_device()

        from safeds.data.image.containers._empty_image_list import _EmptyImageList

        if len(images) == 0:
            return _EmptyImageList()

        image_list = _SingleSizeImageList()
        image_list._tensor_positions_to_indices = []
        images_with_channels: dict[int, list[Tensor]] = {}
        indices_with_channels: dict[int, list[int]] = {}
        max_channel = 0
        for image, index in zip(images, indices, strict=False):
            current_channel = image.size(dim=-3)
            if max_channel < current_channel:
                max_channel = current_channel
            if current_channel not in images_with_channels:
                images_with_channels[current_channel] = [image]
                indices_with_channels[current_channel] = [index]
            else:
                images_with_channels[current_channel].append(image)
                indices_with_channels[current_channel].append(index)

        height = images[0].size(dim=-2)
        width = images[0].size(dim=-1)
        image_list._tensor = torch.empty(len(images), max_channel, height, width, dtype=torch.uint8)
        current_start_index = 0
        for current_channel, ims in images_with_channels.items():
            for index, image in enumerate(ims):
                image_list._tensor[index + current_start_index, 0 : max(current_channel, min(max_channel, 3))] = image
            if max_channel == 4 and current_channel != 4:
                torch.full(
                    (len(ims), 1, height, width),
                    255,
                    out=image_list._tensor[current_start_index : current_start_index + len(ims), 3:4],
                )
            current_start_index += len(ims)
            image_list._tensor_positions_to_indices += indices_with_channels[current_channel]
        image_list._indices_to_tensor_positions = image_list._calc_new_indices_to_tensor_positions()

        return image_list

    @staticmethod
    def _create_from_tensor(images_tensor: Tensor, indices: list[int]) -> _SingleSizeImageList:
        if images_tensor.dim() == 3:
            images_tensor = images_tensor.unsqueeze(dim=1)
        if images_tensor.dim() != 4:
            raise ValueError(f"Invalid Tensor. This Tensor requires 3 or 4 dimensions but has {images_tensor.dim()}")

        image_list = _SingleSizeImageList()
        image_list._tensor = images_tensor.detach().clone()
        image_list._tensor_positions_to_indices = indices
        image_list._indices_to_tensor_positions = image_list._calc_new_indices_to_tensor_positions()

        return image_list

    def __iter__(self) -> _SingleSizeImageList:
        im_ds = copy.copy(self)
        im_ds._next_batch_index = 0
        return im_ds

    def __next__(self) -> Tensor:
        if self._next_batch_index * self._batch_size >= len(self):
            raise StopIteration
        self._next_batch_index += 1
        return self._get_batch(self._next_batch_index - 1)

    def _get_batch(self, batch_number: int, batch_size: int | None = None) -> Tensor:
        import torch

        _init_default_device()

        if batch_size is None:
            batch_size = self._batch_size
        if batch_size * batch_number >= len(self):
            raise IndexOutOfBoundsError(batch_size * batch_number)
        max_index = batch_size * (batch_number + 1) if batch_size * (batch_number + 1) < len(self) else len(self)
        return (
            self._tensor[
                [self._indices_to_tensor_positions[index] for index in range(batch_size * batch_number, max_index)]
            ].to(torch.float32)
            / 255
        )

    def _clone(self) -> ImageList:
        cloned_image_list = self._clone_without_tensor()
        cloned_image_list._tensor = self._tensor.detach().clone()
        return cloned_image_list

    def _clone_without_tensor(self) -> _SingleSizeImageList:
        """
        Clone this SingleSizeImageList to a new instance without the image data.

        Returns
        -------
        image_list:
            the cloned image list
        """
        cloned_image_list = _SingleSizeImageList()
        cloned_image_list._indices_to_tensor_positions = copy.deepcopy(self._indices_to_tensor_positions)
        cloned_image_list._tensor_positions_to_indices = copy.deepcopy(self._tensor_positions_to_indices)
        return cloned_image_list

    def _calc_new_indices_to_tensor_positions(self) -> dict[int, int]:
        """
        Calculate the new indices to tensor position dictionary.

        Returns
        -------
        new_indices_to_tensor_positions:
            the new dictionary
        """
        _indices_to_tensor_positions: dict[int, int] = {}
        for i, index in enumerate(self._tensor_positions_to_indices):
            _indices_to_tensor_positions[index] = i
        return _indices_to_tensor_positions

    def __eq__(self, other: object) -> bool:
        import torch

        _init_default_device()

        if not isinstance(other, ImageList):
            return NotImplemented
        if not isinstance(other, _SingleSizeImageList):
            return False
        return (self is other) or (
            self._tensor.size() == other._tensor.size()
            and set(self._tensor_positions_to_indices) == set(self._tensor_positions_to_indices)
            and set(self._indices_to_tensor_positions) == set(self._indices_to_tensor_positions)
            and torch.all(
                torch.eq(
                    self._tensor[torch.tensor(self._tensor_positions_to_indices).argsort()],
                    other._tensor[torch.tensor(other._tensor_positions_to_indices).argsort()],
                ),
            ).item()
        )

    def __hash__(self) -> int:
        return _structural_hash(
            self.widths[0],
            self.heights[0],
            self.channel,
            self.image_count,
            self._tensor_positions_to_indices,
        )

    def __sizeof__(self) -> int:
        return (
            sys.getsizeof(self._tensor)
            + self._tensor.element_size() * self._tensor.nelement()
            + sum(map(sys.getsizeof, self._tensor_positions_to_indices))
            + sys.getsizeof(self._tensor_positions_to_indices)
            + sum(map(sys.getsizeof, self._indices_to_tensor_positions.keys()))
            + sum(map(sys.getsizeof, self._indices_to_tensor_positions.values()))
            + sys.getsizeof(self._indices_to_tensor_positions)
        )

    @property
    def image_count(self) -> int:
        return self._tensor.size(dim=0)

    @property
    def widths(self) -> list[int]:
        return [self._tensor.size(dim=3)] * self.image_count

    @property
    def heights(self) -> list[int]:
        return [self._tensor.size(dim=2)] * self.image_count

    @property
    def channel(self) -> int:
        return self._tensor.size(dim=1)

    @property
    def sizes(self) -> list[ImageSize]:
        return [
            ImageSize(self._tensor.size(dim=3), self._tensor.size(dim=2), self._tensor.size(dim=1)),
        ] * self.image_count

    @property
    def size_count(self) -> int:
        return 1

    def get_image(self, index: int) -> Image:
        if index not in self._indices_to_tensor_positions:
            raise IndexOutOfBoundsError(index)
        return Image(self._tensor[self._indices_to_tensor_positions[index]])

    def index(self, image: Image) -> list[int]:
        if image not in self:
            return []
        return [
            self._tensor_positions_to_indices[i]
            for i in (image._image_tensor == self._tensor).all(dim=[1, 2, 3]).nonzero()
        ]

    def has_image(self, image: Image) -> bool:
        return (
            not (image.width != self.widths[0] or image.height != self.heights[0] or image.channel != self.channel)
            and image._image_tensor in self._tensor
        )

    def to_jpeg_files(self, path: str | Path | list[str | Path]) -> None:
        import torch
        from torchvision.transforms.v2 import functional as func2
        from torchvision.utils import save_image

        _init_default_device()

        if self.channel == 4:
            raise IllegalFormatError("png")
        path_str: str | Path
        if isinstance(path, list):
            if len(path) == self.image_count:
                for image_path, index in zip(path, sorted(self._tensor_positions_to_indices), strict=False):
                    Path(image_path).parent.mkdir(parents=True, exist_ok=True)
                    if self.channel == 1:
                        func2.to_pil_image(self._tensor[self._indices_to_tensor_positions[index]], mode="L").save(
                            image_path,
                            format="jpeg",
                        )
                    else:
                        save_image(
                            self._tensor[self._indices_to_tensor_positions[index]].to(torch.float32) / 255,
                            image_path,
                            format="jpeg",
                        )
                return
            elif len(path) == 1:
                path_str = path[0]
            else:
                raise ValueError(
                    "The path specified is invalid. Please provide either the path to a directory, a list of paths with one path for each image, or a list of paths with one path per image size.",
                )
        else:
            path_str = path
        for index in self._tensor_positions_to_indices:
            image_path = Path(path_str) / (str(index) + ".jpg")
            Path(image_path).parent.mkdir(parents=True, exist_ok=True)
            if self.channel == 1:
                func2.to_pil_image(self._tensor[self._indices_to_tensor_positions[index]], mode="L").save(
                    image_path,
                    format="jpeg",
                )
            else:
                save_image(
                    self._tensor[self._indices_to_tensor_positions[index]].to(torch.float32) / 255,
                    image_path,
                    format="jpeg",
                )

    def to_png_files(self, path: str | Path | list[str | Path]) -> None:
        import torch
        from torchvision.transforms.v2 import functional as func2
        from torchvision.utils import save_image

        _init_default_device()

        path_str: str | Path
        if isinstance(path, list):
            if len(path) == self.image_count:
                for image_path, index in zip(path, sorted(self._tensor_positions_to_indices), strict=False):
                    Path(image_path).parent.mkdir(parents=True, exist_ok=True)
                    if self.channel == 1:
                        func2.to_pil_image(self._tensor[self._indices_to_tensor_positions[index]], mode="L").save(
                            image_path,
                            format="png",
                        )
                    else:
                        save_image(
                            self._tensor[self._indices_to_tensor_positions[index]].to(torch.float32) / 255,
                            image_path,
                            format="png",
                        )
                return
            elif len(path) == 1:
                path_str = path[0]
            else:
                raise ValueError(
                    "The path specified is invalid. Please provide either the path to a directory, a list of paths with one path for each image, or a list of paths with one path per image size.",
                )
        else:
            path_str = path
        for index in self._tensor_positions_to_indices:
            image_path = Path(path_str) / (str(index) + ".png")
            Path(image_path).parent.mkdir(parents=True, exist_ok=True)
            if self.channel == 1:
                func2.to_pil_image(self._tensor[self._indices_to_tensor_positions[index]], mode="L").save(
                    image_path,
                    format="png",
                )
            else:
                save_image(
                    self._tensor[self._indices_to_tensor_positions[index]].to(torch.float32) / 255,
                    image_path,
                    format="png",
                )

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
        if channel == self.channel:
            return self
        image_list = self._clone_without_tensor()
        image_list._tensor = _SingleSizeImageList._change_channel_of_tensor(self._tensor, channel)
        return image_list

    @staticmethod
    def _change_channel_of_tensor(tensor: Tensor, channel: int) -> Tensor:
        """
        Change the channel of a tensor to the given channel.

        Parameters
        ----------
        tensor:
            the tensor to change the channel of
        channel:
            the given new channel

        Returns
        -------
        new_tensor:
            a new tensor with the given channel

        Raises
        ------
        ValueError
            if the given channel is not a valid channel option
        """
        import torch

        _init_default_device()

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
        import torch

        _init_default_device()

        from safeds.data.image.containers._multi_size_image_list import _MultiSizeImageList

        if index in self._indices_to_tensor_positions:
            raise DuplicateIndexError(index)

        if self._tensor.size(dim=2) == image_tensor.size(dim=1) and self._tensor.size(dim=3) == image_tensor.size(
            dim=2,
        ):
            image_list_single: _SingleSizeImageList = self._clone_without_tensor()
            if image_tensor.size(dim=0) > self.channel:
                tensor = _SingleSizeImageList._change_channel_of_tensor(self._tensor, image_tensor.size(dim=0))
            else:
                tensor = self._tensor
            if image_tensor.size(dim=0) < tensor.size(dim=1):
                if tensor.size(dim=1) == 3:  # image_tensor channel == 1
                    image_list_single._tensor = torch.cat(
                        [tensor, torch.stack([image_tensor, image_tensor, image_tensor], dim=1)],
                    )
                elif image_tensor.size(dim=0) == 1:  # tensor channel == 4
                    image_list_single._tensor = torch.cat(
                        [
                            tensor,
                            torch.stack(
                                [image_tensor, image_tensor, image_tensor, torch.full(image_tensor.size(), 255)],
                                dim=1,
                            ),
                        ],
                    )
                else:  # image_tensor channel == 3; tensor channel == 4
                    image_list_single._tensor = torch.cat(
                        [
                            tensor,
                            torch.cat([image_tensor, torch.full(image_tensor[0:1].size(), 255)], dim=0).unsqueeze(
                                dim=0,
                            ),
                        ],
                    )
            else:
                image_list_single._tensor = torch.cat([tensor, image_tensor.unsqueeze(dim=0)])
            image_list_single._tensor_positions_to_indices.append(index)
            image_list_single._indices_to_tensor_positions[index] = len(self)

            return image_list_single
        else:
            image_list_multi: _MultiSizeImageList = _MultiSizeImageList()
            image_list_multi._image_list_dict[(self.widths[0], self.heights[0])] = self
            image_list_multi._image_list_dict[(image_tensor.size(dim=2), image_tensor.size(dim=1))] = (
                _SingleSizeImageList._create_image_list([image_tensor], [index])
            )

            if image_tensor.size(dim=0) != self.channel:
                image_list_multi = image_list_multi.change_channel(
                    max(image_tensor.size(dim=0), self.channel),
                )._as_multi_size_image_list()
            for _index in self._tensor_positions_to_indices:
                image_list_multi._indices_to_image_size_dict[_index] = (self.widths[0], self.heights[0])
            image_list_multi._indices_to_image_size_dict[index] = (image_tensor.size(dim=2), image_tensor.size(dim=1))

            return image_list_multi

    def add_images(self, images: list[Image] | ImageList) -> ImageList:
        import torch

        _init_default_device()

        from safeds.data.image.containers._empty_image_list import _EmptyImageList
        from safeds.data.image.containers._multi_size_image_list import _MultiSizeImageList

        if isinstance(images, _EmptyImageList) or isinstance(images, list) and len(images) == 0:
            return self

        next_index = max(self._tensor_positions_to_indices) + 1

        max_channel = self.channel
        self_size = (self.widths[0], self.heights[0])

        if isinstance(images, list):
            new_image_lists: dict[tuple[int, int], ImageList] = {}

            images_with_sizes_with_channel: dict[tuple[int, int], dict[int, list[Image]]] = {}
            images_with_sizes_count: dict[tuple[int, int], int] = {}
            indices_with_sizes_with_channel: dict[tuple[int, int], dict[int, list[int]]] = {}
            for image in images:
                current_size = (image.width, image.height)
                current_channel = image.channel
                if max_channel < current_channel:
                    max_channel = current_channel
                if current_size not in images_with_sizes_with_channel:
                    images_with_sizes_with_channel[current_size] = {}
                    indices_with_sizes_with_channel[current_size] = {}
                    images_with_sizes_count[current_size] = 1
                else:
                    images_with_sizes_count[current_size] += 1
                if current_channel not in images_with_sizes_with_channel[current_size]:
                    images_with_sizes_with_channel[current_size][current_channel] = [image]
                    indices_with_sizes_with_channel[current_size][current_channel] = [next_index]
                else:
                    images_with_sizes_with_channel[current_size][current_channel].append(image)
                    indices_with_sizes_with_channel[current_size][current_channel].append(next_index)
                next_index += 1
            if self_size not in images_with_sizes_with_channel:
                if self.channel != max_channel:
                    new_image_lists[self_size] = self.change_channel(max_channel)
                else:
                    new_image_lists[self_size] = self
            for size in images_with_sizes_with_channel:
                if size == self_size:
                    new_tensor = torch.empty(
                        len(self) + images_with_sizes_count[size],
                        max_channel,
                        size[1],
                        size[0],
                        dtype=torch.uint8,
                    )
                    new_indices = self._tensor_positions_to_indices
                    if self.channel != max_channel:
                        new_tensor[0 : len(self)] = _SingleSizeImageList._change_channel_of_tensor(
                            self._tensor,
                            max_channel,
                        )
                    else:
                        new_tensor[0 : len(self)] = self._tensor
                    current_index = len(self)
                    for channel in images_with_sizes_with_channel[size]:
                        new_indices += indices_with_sizes_with_channel[size][channel]
                        number_in_current_channel = len(images_with_sizes_with_channel[size][channel])
                        end_index = current_index + number_in_current_channel
                        if channel < max_channel:
                            if channel == 3:
                                torch.stack(
                                    [img._image_tensor for img in images_with_sizes_with_channel[size][channel]],
                                    dim=0,
                                    out=new_tensor[current_index:end_index, 0:3],
                                )
                            else:  # channel == 1
                                torch.stack(
                                    [img._image_tensor for img in images_with_sizes_with_channel[size][channel]],
                                    dim=0,
                                    out=new_tensor[current_index:end_index, 0:1],
                                )
                                torch.stack(
                                    [img._image_tensor for img in images_with_sizes_with_channel[size][channel]],
                                    dim=0,
                                    out=new_tensor[current_index:end_index, 1:2],
                                )
                                torch.stack(
                                    [img._image_tensor for img in images_with_sizes_with_channel[size][channel]],
                                    dim=0,
                                    out=new_tensor[current_index:end_index, 2:3],
                                )
                            if max_channel == 4:
                                torch.full(
                                    (number_in_current_channel, 1, size[1], size[0]),
                                    255,
                                    dtype=torch.uint8,
                                    out=new_tensor[current_index:end_index, 3:4],
                                )
                        else:
                            torch.stack(
                                [img._image_tensor for img in images_with_sizes_with_channel[size][channel]],
                                dim=0,
                                out=new_tensor[current_index:end_index, :],
                            )
                        current_index = end_index
                    new_image_list = _SingleSizeImageList()
                    new_image_list._tensor = new_tensor
                    new_image_list._tensor_positions_to_indices = new_indices
                    new_image_list._indices_to_tensor_positions = new_image_list._calc_new_indices_to_tensor_positions()
                    new_image_lists[size] = new_image_list
                else:
                    new_tensor = torch.empty(
                        images_with_sizes_count[size],
                        max_channel,
                        size[1],
                        size[0],
                        dtype=torch.uint8,
                    )
                    new_indices = []
                    current_index = 0
                    for channel in images_with_sizes_with_channel[size]:
                        new_indices += indices_with_sizes_with_channel[size][channel]
                        number_in_current_channel = len(images_with_sizes_with_channel[size][channel])
                        end_index = current_index + number_in_current_channel
                        if channel < max_channel:
                            if channel == 3:
                                torch.stack(
                                    [img._image_tensor for img in images_with_sizes_with_channel[size][channel]],
                                    dim=0,
                                    out=new_tensor[current_index:end_index, 0:3],
                                )
                            else:  # channel == 1
                                torch.stack(
                                    [img._image_tensor for img in images_with_sizes_with_channel[size][channel]],
                                    dim=0,
                                    out=new_tensor[current_index:end_index, 0:1],
                                )
                                torch.stack(
                                    [img._image_tensor for img in images_with_sizes_with_channel[size][channel]],
                                    dim=0,
                                    out=new_tensor[current_index:end_index, 1:2],
                                )
                                torch.stack(
                                    [img._image_tensor for img in images_with_sizes_with_channel[size][channel]],
                                    dim=0,
                                    out=new_tensor[current_index:end_index, 2:3],
                                )
                            if max_channel == 4:
                                torch.full(
                                    (number_in_current_channel, 1, size[1], size[0]),
                                    255,
                                    dtype=torch.uint8,
                                    out=new_tensor[current_index:end_index, 3:4],
                                )
                        else:
                            torch.stack(
                                [img._image_tensor for img in images_with_sizes_with_channel[size][channel]],
                                dim=0,
                                out=new_tensor[current_index:end_index, :],
                            )
                        current_index = end_index
                    new_image_list = _SingleSizeImageList()
                    new_image_list._tensor = new_tensor
                    new_image_list._tensor_positions_to_indices = new_indices
                    new_image_list._indices_to_tensor_positions = new_image_list._calc_new_indices_to_tensor_positions()
                    new_image_lists[size] = new_image_list
            if len(new_image_lists) == 1:
                return new_image_lists[next(iter(new_image_lists))]
            else:
                multi_image_list = _MultiSizeImageList()
                multi_image_list._image_list_dict = new_image_lists
                multi_image_list._indices_to_image_size_dict = {}
                for size, im_list in new_image_lists.items():
                    for index in im_list._as_single_size_image_list()._tensor_positions_to_indices:
                        multi_image_list._indices_to_image_size_dict[index] = size
                return multi_image_list
        else:  # images is of type ImageList
            max_channel = max(max_channel, images.channel)
            index_offset = max(self._tensor_positions_to_indices)
            if isinstance(images, _SingleSizeImageList):
                return _SingleSizeImageList._combine_two_single_size_image_lists(self, images)
            else:  # images is of type _MultiSizeImageList
                images = images._as_multi_size_image_list()
                multi_image_list = _MultiSizeImageList()
                multi_image_list._image_list_dict = {}
                multi_image_list._indices_to_image_size_dict = {}
                if self_size in images._image_list_dict:
                    new_self_im_list = _SingleSizeImageList._combine_two_single_size_image_lists(
                        self,
                        images._image_list_dict[self_size]._as_single_size_image_list(),
                    )
                elif self.channel != max_channel:
                    new_self_im_list = self.change_channel(max_channel)
                else:
                    new_self_im_list = self
                multi_image_list._image_list_dict[self_size] = new_self_im_list
                for index in new_self_im_list._as_single_size_image_list()._tensor_positions_to_indices:
                    multi_image_list._indices_to_image_size_dict[index] = self_size
                for im_size, im_list in images._image_list_dict.items():
                    if im_size == self_size:
                        continue
                    if im_list.channel != max_channel:
                        new_im_list = im_list.change_channel(max_channel)._as_single_size_image_list()
                    else:
                        new_im_list = im_list._as_single_size_image_list()
                    new_im_list._tensor_positions_to_indices = [
                        old_index + index_offset
                        for old_index in im_list._as_single_size_image_list()._tensor_positions_to_indices
                    ]
                    new_im_list._indices_to_tensor_positions = new_im_list._calc_new_indices_to_tensor_positions()
                    multi_image_list._image_list_dict[im_size] = new_im_list
                    for new_index in new_im_list._tensor_positions_to_indices:
                        multi_image_list._indices_to_image_size_dict[new_index] = im_size
                return multi_image_list

    @staticmethod
    def _combine_two_single_size_image_lists(
        image_list_1: _SingleSizeImageList,
        image_list_2: _SingleSizeImageList,
    ) -> ImageList:
        import torch

        from safeds.data.image.containers._multi_size_image_list import _MultiSizeImageList

        max_channel = max(image_list_1.channel, image_list_2.channel)
        im1_size = (image_list_1.widths[0], image_list_1.heights[0])
        im2_size = (image_list_2.widths[0], image_list_2.heights[0])
        index_offset = max(image_list_1._tensor_positions_to_indices) + 1

        if im1_size == im2_size:
            if image_list_2.channel == image_list_1.channel:
                new_image_list = _SingleSizeImageList()
                new_image_list._tensor = torch.cat([image_list_1._tensor, image_list_2._tensor], dim=0)
                new_image_list._tensor_positions_to_indices = image_list_1._tensor_positions_to_indices + [
                    old_index + index_offset for old_index in image_list_2._tensor_positions_to_indices
                ]
                new_image_list._indices_to_tensor_positions = new_image_list._calc_new_indices_to_tensor_positions()
                return new_image_list
            else:
                new_image_list = _SingleSizeImageList()
                if image_list_2.channel < max_channel:
                    new_image_list._tensor = torch.cat(
                        [
                            image_list_1._tensor,
                            _SingleSizeImageList._change_channel_of_tensor(image_list_2._tensor, max_channel),
                        ],
                        dim=0,
                    )
                else:
                    new_image_list._tensor = torch.cat(
                        [
                            _SingleSizeImageList._change_channel_of_tensor(image_list_1._tensor, max_channel),
                            image_list_2._tensor,
                        ],
                        dim=0,
                    )
                new_image_list._tensor_positions_to_indices = image_list_1._tensor_positions_to_indices + [
                    old_index + index_offset for old_index in image_list_2._tensor_positions_to_indices
                ]
                new_image_list._indices_to_tensor_positions = new_image_list._calc_new_indices_to_tensor_positions()
                return new_image_list
        else:
            multi_image_list = _MultiSizeImageList()
            multi_image_list._image_list_dict = {}
            im_l_2 = _SingleSizeImageList()
            if image_list_2.channel == image_list_1.channel:
                multi_image_list._image_list_dict[im1_size] = image_list_1
                im_l_2._tensor = image_list_2._tensor
            elif image_list_2.channel < image_list_1.channel:
                multi_image_list._image_list_dict[im1_size] = image_list_1
                im_l_2._tensor = _SingleSizeImageList._change_channel_of_tensor(image_list_2._tensor, max_channel)
            else:
                multi_image_list._image_list_dict[im1_size] = image_list_1.change_channel(max_channel)
                im_l_2._tensor = image_list_2._tensor
            im_l_2._tensor_positions_to_indices = [
                old_index + index_offset for old_index in image_list_2._tensor_positions_to_indices
            ]
            im_l_2._indices_to_tensor_positions = im_l_2._calc_new_indices_to_tensor_positions()
            multi_image_list._image_list_dict[im2_size] = im_l_2
            multi_image_list._indices_to_image_size_dict = {}
            for index in image_list_1._tensor_positions_to_indices:
                multi_image_list._indices_to_image_size_dict[index] = im1_size
            for index in image_list_2._tensor_positions_to_indices:
                multi_image_list._indices_to_image_size_dict[index + index_offset] = im2_size
            return multi_image_list

    def remove_image_by_index(self, index: int | list[int]) -> ImageList:
        if isinstance(index, int):
            index = [index]

        invalid_indices = []
        for _i in index:
            if _i not in self._tensor_positions_to_indices:
                invalid_indices.append(_i)
        if len(invalid_indices) > 0:
            raise IndexOutOfBoundsError(invalid_indices)

        return self._remove_image_by_index_ignore_invalid(index)

    def _remove_image_by_index_ignore_invalid(self, index: int | list[int]) -> ImageList:
        from safeds.data.image.containers._empty_image_list import _EmptyImageList

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
        image_list._tensor = self._tensor[
            [i for i, v in enumerate(self._tensor_positions_to_indices) if v not in index]
        ]
        image_list._tensor_positions_to_indices = [
            i - len([k for k in index if k < i]) for i in self._tensor_positions_to_indices if i not in index
        ]
        image_list._indices_to_tensor_positions = image_list._calc_new_indices_to_tensor_positions()
        return image_list

    def remove_images_with_size(self, width: int, height: int) -> ImageList:
        from safeds.data.image.containers._empty_image_list import _EmptyImageList

        _check_remove_images_with_size_errors(width, height)

        if self.widths[0] == width and self.heights[0] == height:
            return _EmptyImageList()
        else:
            return self

    def remove_duplicate_images(self) -> ImageList:
        image_list = _SingleSizeImageList()
        tensor_cpu_unique, new_indices = self._tensor.cpu().unique(
            dim=0,
            return_inverse=True,
        )  # Works somehow faster on cpu
        if tensor_cpu_unique.size(dim=-4) == self._tensor.size(dim=-4):  # no duplicates
            return self
        image_list._tensor = tensor_cpu_unique.to(self._tensor.device)
        indices, indices_to_remove = [], []
        offset_indices: list[int] = []
        for index_n, index_v in enumerate(new_indices):
            if list(new_indices).index(index_v) == index_n:
                indices.append(
                    self._tensor_positions_to_indices[
                        index_v + len([offset for offset in offset_indices if index_v > offset])
                    ],
                )
            else:
                offset_indices.append(index_v)
        indices_to_remove = [index_rem for index_rem in self._tensor_positions_to_indices if index_rem not in indices]
        for index_r in indices_to_remove:
            indices = [index - 1 if index > index_r else index for index in indices]
        image_list._tensor_positions_to_indices = indices
        image_list._indices_to_tensor_positions = image_list._calc_new_indices_to_tensor_positions()
        return image_list

    def shuffle_images(self) -> ImageList:
        image_list = self._clone()._as_single_size_image_list()
        random.shuffle(image_list._tensor_positions_to_indices)
        image_list._indices_to_tensor_positions = image_list._calc_new_indices_to_tensor_positions()
        return image_list

    def resize(self, new_width: int, new_height: int) -> ImageList:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2 import functional as func2

        _init_default_device()

        _check_resize_errors(new_width, new_height)
        image_list = self._clone_without_tensor()
        image_list._tensor = func2.resize(
            self._tensor,
            size=[new_height, new_width],
            interpolation=InterpolationMode.NEAREST,
        )
        return image_list

    def convert_to_grayscale(self) -> ImageList:
        if self.channel == 1:
            return self
        image_list = self._clone_without_tensor()
        image_list._tensor = _SingleSizeImageList._convert_tensor_to_grayscale(self._tensor)
        return image_list

    @staticmethod
    def _convert_tensor_to_grayscale(tensor: Tensor) -> Tensor:
        import torch
        from torchvision.transforms.v2 import functional as func2

        _init_default_device()

        if tensor.size(dim=-3) == 1:
            return tensor
        elif tensor.size(dim=-3) == 4:
            return torch.cat(
                [func2.rgb_to_grayscale(tensor[:, 0:3], num_output_channels=3), tensor[:, 3:4]],
                dim=1,
            )
        else:  # channel == 3
            return func2.rgb_to_grayscale(tensor[:, 0:3], num_output_channels=3)

    def crop(self, x: int, y: int, width: int, height: int) -> ImageList:
        from torchvision.transforms.v2 import functional as func2

        _init_default_device()

        _check_crop_errors(x, y, width, height)
        _check_crop_warnings(x, y, self.widths[0], self.heights[0], plural=True)
        image_list = self._clone_without_tensor()
        image_list._tensor = func2.crop(self._tensor, x, y, height, width)
        return image_list

    def flip_vertically(self) -> ImageList:
        from torchvision.transforms.v2 import functional as func2

        _init_default_device()

        image_list = self._clone_without_tensor()
        image_list._tensor = func2.vertical_flip(self._tensor)
        return image_list

    def flip_horizontally(self) -> ImageList:
        from torchvision.transforms.v2 import functional as func2

        _init_default_device()

        image_list = self._clone_without_tensor()
        image_list._tensor = func2.horizontal_flip(self._tensor)
        return image_list

    def adjust_brightness(self, factor: float) -> ImageList:
        import torch

        _init_default_device()

        _check_adjust_brightness_errors_and_warnings(factor, plural=True)
        image_list = self._clone_without_tensor()
        image_list._tensor = torch.empty(self._tensor.size(), dtype=torch.uint8)
        channel = self._tensor.size(dim=-3)
        if factor == 0:
            torch.zeros(
                (self._tensor.size(dim=-4), min(3, channel), self._tensor.size(dim=-2), self._tensor.size(dim=-1)),
                dtype=torch.uint8,
                out=image_list._tensor[:, 0 : min(self._tensor.size(dim=-3), 3)],
            )
        else:
            temp_tensor = self._tensor[:, 0 : min(channel, 3)] * torch.tensor([factor * 1.0], dtype=torch.float16)
            torch.clamp(temp_tensor, 0, 255, out=temp_tensor)
            image_list._tensor[:, 0 : min(channel, 3)] = temp_tensor[:, :]
        if channel == 4:
            image_list._tensor[:, 3] = self._tensor[:, 3]

        return image_list

    def add_noise(self, standard_deviation: float) -> ImageList:
        import torch

        _init_default_device()

        _check_add_noise_errors(standard_deviation)
        image_list = self._clone_without_tensor()
        image_list._tensor = torch.empty(self._tensor.size(), dtype=torch.uint8)
        float_tensor = torch.empty(self._tensor.size(), dtype=torch.float16)
        torch.normal(0, standard_deviation, self._tensor.size(), out=float_tensor)
        float_tensor *= 255
        float_tensor += self._tensor
        torch.clamp(float_tensor, 0, 255, out=float_tensor)
        image_list._tensor[:] = float_tensor[:]
        return image_list

    def adjust_contrast(self, factor: float) -> ImageList:
        import torch

        _init_default_device()

        _check_adjust_contrast_errors_and_warnings(factor, plural=True)
        image_list = self._clone_without_tensor()
        image_list._tensor = torch.empty(self._tensor.size(), dtype=torch.uint8)

        channel = self._tensor.size(dim=-3)
        factor *= 1.0
        adjusted_factor = (1 - factor) / factor

        gray_tensor = _SingleSizeImageList._convert_tensor_to_grayscale(self._tensor[:, 0 : min(channel, 3)])
        mean = torch.mean(gray_tensor, dim=(-3, -2, -1), keepdim=True, dtype=torch.float16)
        del gray_tensor
        mean *= torch.tensor(adjusted_factor, dtype=torch.float16)
        adjusted_tensor = mean.repeat(1, min(channel, 3), self._tensor.size(dim=-2), self._tensor.size(dim=-1))
        adjusted_tensor += self._tensor[:, 0 : min(channel, 3)]
        adjusted_tensor *= factor
        torch.clamp(adjusted_tensor, 0, 255, out=adjusted_tensor)
        image_list._tensor[:, 0 : min(channel, 3)] = adjusted_tensor[:, :]

        if channel == 4:
            image_list._tensor[:, 3] = self._tensor[:, 3]

        return image_list

    def adjust_color_balance(self, factor: float) -> ImageList:
        import torch

        _check_adjust_color_balance_errors_and_warnings(factor, self.channel, plural=True)
        image_list = self._clone_without_tensor()
        factor *= 1.0
        if factor == 0:
            image_list._tensor = _SingleSizeImageList._convert_tensor_to_grayscale(self._tensor)
        else:
            adjusted_factor = (1 - factor) / factor
            image_list._tensor = torch.empty(self._tensor.size(), dtype=torch.uint8)
            adjusted_tensor = _SingleSizeImageList._convert_tensor_to_grayscale(self._tensor) * torch.tensor(
                adjusted_factor,
                dtype=torch.float16,
            )
            adjusted_tensor += self._tensor
            adjusted_tensor *= factor
            torch.clamp(adjusted_tensor, 0, 255, out=adjusted_tensor)
            image_list._tensor[:] = adjusted_tensor[:]
        return image_list

    def blur(self, radius: int) -> ImageList:
        import torch

        _init_default_device()

        float_dtype = torch.float32 if _get_device() != torch.device("cuda") else torch.float16

        _check_blur_errors_and_warnings(radius, min(self.widths[0], self.heights[0]), plural=True)
        image_list = self._clone_without_tensor()

        image_list._tensor = torch.empty(self._tensor.size(), dtype=torch.uint8)

        kernel = torch.full(
            (self._tensor.size(dim=-3), 1, radius * 2 + 1, radius * 2 + 1),
            1 / (radius * 2 + 1) ** 2,
            dtype=float_dtype,
        )
        image_tensor_size = (
            self._tensor.size(dim=1) * (self._tensor.size(dim=2) + radius * 2) * (self._tensor.size(dim=3) + radius * 2)
        )
        number_of_executions = math.ceil(self._tensor.size(dim=0) / (2**31 / image_tensor_size))
        number_of_images_per_execution = math.ceil(self._tensor.size(dim=0) / number_of_executions)
        start = 0
        for i in range(number_of_executions):
            end = min((i + 1) * number_of_images_per_execution, self._tensor.size(dim=0)) + 1
            image_list._tensor[start:end] = torch.nn.functional.conv2d(
                torch.nn.functional.pad(
                    self._tensor[start:end].to(float_dtype),
                    (radius, radius, radius, radius),
                    mode="replicate",
                ),
                kernel,
                padding="valid",
                groups=self._tensor.size(dim=-3),
            )[:]
            start = end
        return image_list

    def sharpen(self, factor: float) -> ImageList:
        import torch
        from torchvision.transforms.v2 import functional as func2

        _init_default_device()

        _check_sharpen_errors_and_warnings(factor, plural=True)
        image_list = self._clone_without_tensor()
        image_list._tensor = torch.empty(self._tensor.size(), dtype=torch.uint8)

        image_tensor_size = self._tensor.size(dim=1) * self._tensor.size(dim=2) * self._tensor.size(dim=3)
        number_of_executions = math.ceil(self._tensor.size(dim=0) / (2**31 / image_tensor_size))
        number_of_images_per_execution = math.ceil(self._tensor.size(dim=0) / number_of_executions)
        start = 0

        if self.channel == 4:
            for i in range(number_of_executions):
                end = min((i + 1) * number_of_images_per_execution, self._tensor.size(dim=0)) + 1
                image_list._tensor[start:end] = torch.cat(
                    [func2.adjust_sharpness(self._tensor[start:end, 0:3], factor * 1.0), self._tensor[start:end, 3:4]],
                    dim=1,
                )
                start = end
        else:
            for i in range(number_of_executions):
                end = min((i + 1) * number_of_images_per_execution, self._tensor.size(dim=0)) + 1
                image_list._tensor[start:end] = func2.adjust_sharpness(self._tensor[start:end], factor * 1.0)
                start = end
        return image_list

    def invert_colors(self) -> ImageList:
        import torch
        from torchvision.transforms.v2 import functional as func2

        _init_default_device()

        image_list = self._clone_without_tensor()
        if self.channel == 4:
            image_list._tensor = torch.cat(
                [func2.invert(self._tensor[:, 0:3]), self._tensor[:, 3:4]],
                dim=1,
            )
        else:
            image_list._tensor = func2.invert(self._tensor)
        return image_list

    def rotate_right(self) -> ImageList:
        from torchvision.transforms.v2 import functional as func2

        _init_default_device()

        image_list = self._clone_without_tensor()
        image_list._tensor = func2.rotate(self._tensor, -90, expand=True)
        return image_list

    def rotate_left(self) -> ImageList:
        from torchvision.transforms.v2 import functional as func2

        _init_default_device()

        image_list = self._clone_without_tensor()
        image_list._tensor = func2.rotate(self._tensor, 90, expand=True)
        return image_list

    def find_edges(self) -> ImageList:
        import torch

        _init_default_device()

        kernel = Image._filter_edges_kernel()

        image_list = self._clone_without_tensor()

        edges_tensor_float16 = torch.nn.functional.conv2d(
            _SingleSizeImageList._convert_tensor_to_grayscale(self._tensor).to(torch.float16)[:, 0:1],
            kernel,
            padding="same",
        )
        torch.clamp(edges_tensor_float16, 0, 255, out=edges_tensor_float16)
        if self.channel == 1:
            image_list._tensor = edges_tensor_float16.to(torch.uint8)
            return image_list
        edges_tensor = edges_tensor_float16.to(torch.uint8)
        del edges_tensor_float16
        if self.channel == 3:
            image_list._tensor = edges_tensor.repeat(1, 3, 1, 1)
        else:  # self.channel == 4
            image_list._tensor = torch.cat(
                [edges_tensor.repeat(1, 3, 1, 1), self._tensor[:, 3:4]],
                dim=1,
            )
        return image_list
