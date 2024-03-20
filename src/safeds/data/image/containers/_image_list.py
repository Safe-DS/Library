from __future__ import annotations

import io
import math
import os
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from PIL.Image import open as pil_image_open
from torch import Tensor
from torchvision.transforms.v2 import PILToTensor
from torchvision.utils import make_grid, save_image

from safeds.data.image.containers import Image

if TYPE_CHECKING:
    from safeds.data.image.containers import _EmptyImageList, _SingleSizeImageList, _MultiSizeImageList


class ImageList(metaclass=ABCMeta):

    _pil_to_tensor = PILToTensor()

    @staticmethod
    @abstractmethod
    def _create_image_list(images: list[Tensor], indices: list[int]) -> ImageList:
        pass

    @staticmethod
    def from_images(images: list[Image], indices: list[int] | None = None) -> ImageList:
        from safeds.data.image.containers import _EmptyImageList, _SingleSizeImageList, _MultiSizeImageList

        if len(images) == 0:
            return _EmptyImageList()
        if indices is None:
            indices = list(range(len(images)))

        first_width = images[0].width
        first_height = images[0].height
        for im in images:
            if first_width != im.width or first_height != im.height:
                return _MultiSizeImageList._create_image_list([image._image_tensor for image in images], indices)
        return _SingleSizeImageList._create_image_list([image._image_tensor for image in images], indices)

    @staticmethod
    def from_files(path: str | Path | list[str | Path], indices: list[int] | None = None) -> ImageList:
        from safeds.data.image.containers import _EmptyImageList, _SingleSizeImageList, _MultiSizeImageList

        if isinstance(path, list) and len(path) == 0:
            return _EmptyImageList()

        image_tensors = []
        fixed_size = True

        if isinstance(path, str) or isinstance(path, Path):
            path = [Path(path)]
        while len(path) != 0:
            p = Path(path.pop(0))
            if p.is_dir():
                path += sorted([os.path.join(p, name) for name in os.listdir(p)])
            else:
                image_tensors.append(ImageList._pil_to_tensor(pil_image_open(p)))
                if fixed_size and (image_tensors[0].size(dim=2) != image_tensors[-1].size(dim=2) or image_tensors[0].size(dim=1) != image_tensors[-1].size(dim=1)):
                    fixed_size = False

        if len(image_tensors) == 0:
            return _EmptyImageList()

        if indices is None:
            indices = list(range(len(image_tensors)))

        if fixed_size:
            return _SingleSizeImageList._create_image_list(image_tensors, indices)
        else:
            return _MultiSizeImageList._create_image_list(image_tensors, indices)

    @abstractmethod
    def clone(self) -> ImageList:
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass

    def __len__(self):
        return self.number_of_images

    def __contains__(self, item):
        return isinstance(item, Image) and self.has_image(item)

    def _repr_png_(self) -> bytes:
        from safeds.data.image.containers import _EmptyImageList

        if isinstance(self, _EmptyImageList):
            raise ValueError("You cannot display an empty ImageList")

        max_width, max_height = max(self.widths), max(self.heights)
        tensors = []
        for image in self.to_images():
            im_tensor = torch.zeros([4, max_height, max_width])
            im_tensor[:, :image.height, :image.width] = image.change_channel(4)._image_tensor
            tensors.append(im_tensor)
        tensor_grid = make_grid(tensors, math.ceil(math.sqrt(len(tensors))))
        buffer = io.BytesIO()
        save_image(tensor_grid.to(torch.float32) / 255, buffer, format="png")
        buffer.seek(0)
        return buffer.read()

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    @abstractmethod
    def number_of_images(self) -> int:
        pass

    @property
    @abstractmethod
    def widths(self) -> list[int]:
        pass

    @property
    @abstractmethod
    def heights(self) -> list[int]:
        pass

    @property
    @abstractmethod
    def channel(self) -> int:
        pass

    @property
    @abstractmethod
    def number_of_sizes(self) -> int:
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def get_image(self, index: int) -> Image:
        pass

    @abstractmethod
    def index(self, image: Image) -> list[int]:
        pass

    @abstractmethod
    def has_image(self, image: Image) -> bool:
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def to_jpeg_files(self, path: str | Path | list[str] | list[Path]) -> None:
        pass

    @abstractmethod
    def to_png_files(self, path: str | Path | list[str] | list[Path]) -> None:
        pass

    @abstractmethod
    def to_images(self, indices: list[int] | None = None) -> list[Image]:
        pass

    def _as_multi_size_image_list(self) -> _MultiSizeImageList:
        from safeds.data.image.containers import _MultiSizeImageList
        if isinstance(self, _MultiSizeImageList):
            return self
        raise ValueError("The given image_list is not a MultiSizeImageList")

    def _as_single_size_image_list(self) -> _SingleSizeImageList:
        from safeds.data.image.containers import _SingleSizeImageList
        if isinstance(self, _SingleSizeImageList):
            return self
        raise ValueError("The given image_list is not a SingleSizeImageList")

    # ------------------------------------------------------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def change_channel(self, channel: int) -> ImageList:
        pass

    @abstractmethod
    def _add_image_tensor(self, image_tensor: Tensor, index: int) -> ImageList:
        pass

    def add_image(self, image: Image) -> ImageList:
        return self._add_image_tensor(image._image_tensor, self.number_of_images)

    @abstractmethod
    def add_images(self, images: list[Image] | ImageList) -> ImageList:
        pass

    def remove_image(self, image: Image) -> ImageList:
        return self.remove_image_by_index(self.index(image))

    def remove_images(self, images: list[Image]) -> ImageList:
        indices_to_remove = []
        for image in images:
            indices_to_remove += self.index(image)
        return self.remove_image_by_index(list(set(indices_to_remove)))

    @abstractmethod
    def remove_image_by_index(self, index: int | list[int]) -> ImageList:
        pass

    @abstractmethod
    def remove_images_with_size(self, width: int, height: int) -> ImageList:
        pass

    @abstractmethod
    def remove_duplicate_images(self) -> ImageList:
        pass

    @abstractmethod
    def shuffle_images(self) -> ImageList:
        pass

    @abstractmethod
    def resize(self, new_width: int, new_height: int) -> ImageList:
        pass

    @abstractmethod
    def convert_to_grayscale(self) -> ImageList:
        pass

    @abstractmethod
    def crop(self, x: int, y: int, width: int, height: int) -> ImageList:
        pass

    @abstractmethod
    def flip_vertically(self) -> ImageList:
        pass

    @abstractmethod
    def flip_horizontally(self) -> ImageList:
        pass

    @abstractmethod
    def adjust_brightness(self, factor: float) -> ImageList:
        pass

    @abstractmethod
    def add_noise(self, standard_deviation: float) -> ImageList:
        pass

    @abstractmethod
    def adjust_contrast(self, factor: float) -> ImageList:
        pass

    @abstractmethod
    def adjust_color_balance(self, factor: float) -> ImageList:
        pass

    @abstractmethod
    def blur(self, radius: int) -> ImageList:
        pass

    @abstractmethod
    def sharpen(self, factor: float) -> ImageList:
        pass

    @abstractmethod
    def invert_colors(self) -> ImageList:
        pass

    @abstractmethod
    def rotate_right(self) -> ImageList:
        pass

    @abstractmethod
    def rotate_left(self) -> ImageList:
        pass

    @abstractmethod
    def find_edges(self) -> ImageList:
        pass
