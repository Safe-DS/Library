from __future__ import annotations

import copy
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from safeds._config import _init_default_device, _get_device
from safeds._utils import _structural_hash
from safeds.data.image._utils._image_transformation_error_and_warning_checks import (
    _check_add_noise_errors,
    _check_adjust_brightness_errors_and_warnings,
    _check_adjust_color_balance_errors_and_warnings,
    _check_adjust_contrast_errors_and_warnings,
    _check_blur_errors_and_warnings,
    _check_crop_errors_and_warnings,
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
        number_of_images: int,
        max_channel: int,
        width: int,
        height: int,
        indices: dict[int, list[int]],
        max_files_per_thread_package: int,
    ) -> tuple[ImageList, list[ImageList._FromFileThreadPackage]]:
        import torch

        _init_default_device()

        from safeds.data.image.containers._empty_image_list import _EmptyImageList

        if len(images) == 0 or number_of_images == 0:
            return _EmptyImageList(), []

        image_list = _SingleSizeImageList()

        images_tensor = torch.empty(
            number_of_images, max_channel, height, width, dtype=torch.uint8, device=_get_device()
        )

        thread_packages: list[ImageList._FromFileThreadPackage] = []
        current_thread_channel: int | None = None
        current_thread_channel_files: list[str] = []
        current_thread_start_index: int = 0
        while number_of_images - current_thread_start_index > 0:
            num_of_files = min(max_files_per_thread_package, number_of_images - current_thread_start_index)
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
        images_ready_to_concat: list[tuple[Tensor, int]] = []
        images_with_correct_channel: list[tuple[Tensor, int]] = []
        images_with_less_channels: list[tuple[Tensor, int]] = []
        max_channel = 0
        for image, index in zip(images, indices, strict=False):
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
                image_to_append = torch.cat([image, image, image], dim=0)
            elif image.size(dim=0) == 1:  # image channel 1 and max channel 4
                image_to_append = torch.cat(
                    [image, image, image, torch.full(image.size(), 255, device=image.device)],
                    dim=0,
                )
            else:  # image channel 3 and max channel 4
                image_to_append = torch.cat([image, torch.full(image[0:1].size(), 255, device=image.device)], dim=0)
            images_ready_to_concat.append((image_to_append.unsqueeze(dim=0), index))
        image_list._tensor = torch.cat([image for image, index in images_ready_to_concat])
        image_list._tensor_positions_to_indices = [index for image, index in images_ready_to_concat]
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
            self.number_of_images,
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
    def sizes(self) -> list[ImageSize]:
        return [
            ImageSize(self._tensor.size(dim=3), self._tensor.size(dim=2), self._tensor.size(dim=1)),
        ] * self.number_of_images

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
            if len(path) == self.number_of_images:
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
            if len(path) == self.number_of_images:
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
        import torch

        _init_default_device()

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
                tensor = self.change_channel(image_tensor.size(dim=0))._as_single_size_image_list()._tensor
            else:
                tensor = self._tensor
            if image_tensor.size(dim=0) < tensor.size(dim=1):
                if tensor.size(dim=1) == 3:
                    image_list_single._tensor = torch.cat(
                        [tensor, torch.cat([image_tensor, image_tensor, image_tensor], dim=0).unsqueeze(dim=0)],
                    )
                elif image_tensor.size(dim=0) == 1:
                    image_list_single._tensor = torch.cat(
                        [
                            tensor,
                            torch.cat(
                                [image_tensor, image_tensor, image_tensor, torch.full(image_tensor.size(), 255)],
                                dim=0,
                            ).unsqueeze(dim=0),
                        ],
                    )
                else:
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

        first_new_index = max(self._tensor_positions_to_indices) + 1
        if isinstance(images, list):
            images = ImageList.from_images(images)
        if isinstance(images, _MultiSizeImageList):
            image_list_multi: _MultiSizeImageList = _MultiSizeImageList()
            image_list_multi._image_list_dict[(self.widths[0], self.heights[0])] = self
            for index in self._tensor_positions_to_indices:
                image_list_multi._indices_to_image_size_dict[index] = (self.widths[0], self.heights[0])
            return image_list_multi.add_images(images)
        else:
            images_as_single_size_image_list: _SingleSizeImageList = images._as_single_size_image_list()
            new_indices = [
                index + first_new_index for index in images_as_single_size_image_list._tensor_positions_to_indices
            ]
            if (
                images_as_single_size_image_list.widths[0] == self.widths[0]
                and images_as_single_size_image_list.heights[0] == self.heights[0]
            ):
                image_list_single: _SingleSizeImageList = self._clone_without_tensor()._as_single_size_image_list()
                image_list_single._tensor_positions_to_indices += new_indices
                if self.channel > images_as_single_size_image_list.channel:
                    image_list_single._tensor = torch.cat(
                        [
                            self._tensor,
                            _SingleSizeImageList._change_channel_of_tensor(
                                images_as_single_size_image_list._tensor,
                                self.channel,
                            ),
                        ],
                    )
                elif self.channel < images_as_single_size_image_list.channel:
                    image_list_single._tensor = torch.cat(
                        [
                            _SingleSizeImageList._change_channel_of_tensor(
                                self._tensor,
                                images_as_single_size_image_list.channel,
                            ),
                            images_as_single_size_image_list._tensor,
                        ],
                    )
                else:
                    image_list_single._tensor = torch.cat([self._tensor, images_as_single_size_image_list._tensor])
                image_list_single._indices_to_tensor_positions = (
                    image_list_single._calc_new_indices_to_tensor_positions()
                )
                return image_list_single
            else:
                image_list_multi = _MultiSizeImageList()
                image_list_multi._image_list_dict[(self.widths[0], self.heights[0])] = self
                image_list_multi._image_list_dict[
                    (images_as_single_size_image_list.widths[0], images_as_single_size_image_list.heights[0])
                ] = images_as_single_size_image_list
                for index in self._tensor_positions_to_indices:
                    image_list_multi._indices_to_image_size_dict[index] = (self.widths[0], self.heights[0])
                for index in images_as_single_size_image_list._tensor_positions_to_indices:
                    image_list_multi._indices_to_image_size_dict[index + first_new_index] = (
                        images_as_single_size_image_list.widths[0],
                        images_as_single_size_image_list.heights[0],
                    )

                if images_as_single_size_image_list.channel != self.channel:
                    image_list_multi = image_list_multi.change_channel(
                        max(images_as_single_size_image_list.channel, self.channel),
                    )._as_multi_size_image_list()
                return image_list_multi

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
        ].clone()
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
        image_list = self._clone_without_tensor()
        image_list._tensor = _SingleSizeImageList._convert_tensor_to_grayscale(self._tensor)
        return image_list

    @staticmethod
    def _convert_tensor_to_grayscale(tensor: Tensor) -> Tensor:
        import torch
        from torchvision.transforms.v2 import functional as func2

        _init_default_device()

        if tensor.size(dim=-3) == 4:
            return torch.cat(
                [func2.rgb_to_grayscale(tensor[:, 0:3], num_output_channels=3), tensor[:, 3].unsqueeze(dim=1)],
                dim=1,
            )
        else:
            return func2.rgb_to_grayscale(tensor[:, 0:3], num_output_channels=tensor.size(dim=-3))

    def crop(self, x: int, y: int, width: int, height: int) -> ImageList:
        from torchvision.transforms.v2 import functional as func2

        _init_default_device()

        _check_crop_errors_and_warnings(x, y, width, height, self.widths[0], self.heights[0], plural=True)
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
        from torchvision.transforms.v2 import functional as func2

        _init_default_device()

        _check_adjust_brightness_errors_and_warnings(factor, plural=True)
        image_list = self._clone_without_tensor()
        if self.channel == 4:
            image_list._tensor = torch.cat(
                [func2.adjust_brightness(self._tensor[:, 0:3], factor * 1.0), self._tensor[:, 3].unsqueeze(dim=1)],
                dim=1,
            )
        else:
            image_list._tensor = func2.adjust_brightness(self._tensor, factor * 1.0)
        return image_list

    def add_noise(self, standard_deviation: float) -> ImageList:
        import torch

        _init_default_device()

        _check_add_noise_errors(standard_deviation)
        image_list = self._clone_without_tensor()
        image_list._tensor = (
            self._tensor + torch.normal(0, standard_deviation, self._tensor.size()).to(_get_device()) * 255
        )
        return image_list

    def adjust_contrast(self, factor: float) -> ImageList:
        import torch
        from torchvision.transforms.v2 import functional as func2

        _init_default_device()

        _check_adjust_contrast_errors_and_warnings(factor, plural=True)
        image_list = self._clone_without_tensor()
        if self.channel == 4:
            image_list._tensor = torch.cat(
                [func2.adjust_contrast(self._tensor[:, 0:3], factor * 1.0), self._tensor[:, 3].unsqueeze(dim=1)],
                dim=1,
            )
        else:
            image_list._tensor = func2.adjust_contrast(self._tensor, factor * 1.0)
        return image_list

    def adjust_color_balance(self, factor: float) -> ImageList:
        _check_adjust_color_balance_errors_and_warnings(factor, self.channel, plural=True)
        image_list = self._clone_without_tensor()
        image_list._tensor = self.convert_to_grayscale()._as_single_size_image_list()._tensor * (
            1.0 - factor * 1.0
        ) + self._tensor * (factor * 1.0)
        return image_list

    def blur(self, radius: int) -> ImageList:
        from torchvision.transforms.v2 import functional as func2

        _init_default_device()

        _check_blur_errors_and_warnings(radius, min(self.widths[0], self.heights[0]), plural=True)
        image_list = self._clone_without_tensor()
        image_list._tensor = func2.gaussian_blur(self._tensor, [radius * 2 + 1, radius * 2 + 1])
        return image_list

    def sharpen(self, factor: float) -> ImageList:
        import torch
        from torchvision.transforms.v2 import functional as func2

        _init_default_device()

        _check_sharpen_errors_and_warnings(factor, plural=True)
        image_list = self._clone_without_tensor()
        if self.channel == 4:
            image_list._tensor = torch.cat(
                [func2.adjust_sharpness(self._tensor[:, 0:3], factor * 1.0), self._tensor[:, 3].unsqueeze(dim=1)],
                dim=1,
            )
        else:
            image_list._tensor = func2.adjust_sharpness(self._tensor, factor * 1.0)
        return image_list

    def invert_colors(self) -> ImageList:
        import torch
        from torchvision.transforms.v2 import functional as func2

        _init_default_device()

        image_list = self._clone_without_tensor()
        if self.channel == 4:
            image_list._tensor = torch.cat(
                [func2.invert(self._tensor[:, 0:3]), self._tensor[:, 3].unsqueeze(dim=1)],
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
            image_list._tensor = torch.cat(
                [edges_tensor.repeat(1, 3, 1, 1), self._tensor[:, 3].unsqueeze(dim=1)],
                dim=1,
            )
        else:
            image_list._tensor = edges_tensor
        return image_list
