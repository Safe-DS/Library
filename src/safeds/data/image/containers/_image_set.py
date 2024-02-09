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
    from safeds.data.image.containers import _FixedSizedImageSet
    from safeds.data.image.containers import _VariousSizedImageSet


class ImageSet(metaclass=ABCMeta):

    _pil_to_tensor = PILToTensor()

    @staticmethod
    @abstractmethod
    def _create_image_set(images: list[Tensor], indices: list[int]) -> ImageSet:
        pass

    @staticmethod
    def from_images(images: list[Image], indices: list[int] | None = None) -> ImageSet:
        from safeds.data.image.containers import _FixedSizedImageSet
        from safeds.data.image.containers import _VariousSizedImageSet

        if indices is None:
            indices = list(range(len(images)))

        first_width = images[0].width
        first_height = images[0].height
        for im in images:
            if first_width != im.width or first_height != im.height:
                return _VariousSizedImageSet._create_image_set([image._image_tensor for image in images], indices)
        return _FixedSizedImageSet._create_image_set([image._image_tensor for image in images], indices)

    @staticmethod
    def from_files(path: str | Path | list[str | Path], indices: list[int] | None = None) -> ImageSet:
        from safeds.data.image.containers import _FixedSizedImageSet
        from safeds.data.image.containers import _VariousSizedImageSet

        image_tensors = []
        fixed_size = True

        if isinstance(path, str) or isinstance(path, Path):
            path = [Path(path)]
        while len(path) != 0:
            p = Path(path.pop(0))
            if p.is_dir():
                path += [str(p) + "\\" + name for name in os.listdir(p)]
            else:
                image_tensors.append(ImageSet._pil_to_tensor(pil_image_open(p)))
                if fixed_size and (image_tensors[0].size(dim=2) != image_tensors[-1].size(dim=2) or image_tensors[0].size(dim=1) != image_tensors[-1].size(dim=1)):
                    fixed_size = False

        if indices is None:
            indices = list(range(len(image_tensors)))

        if fixed_size:
            return _FixedSizedImageSet._create_image_set(image_tensors, indices)
        else:
            return _VariousSizedImageSet._create_image_set(image_tensors, indices)

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass

    def __len__(self):
        return self.number_of_images

    def __contains__(self, item):
        if not isinstance(item, Image):
            return NotImplementedError
        return self.has_image(item)

    def _repr_png_(self) -> bytes:
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

    @property
    @abstractmethod
    def indices(self) -> list[int]:
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

    def _as_various_sized_image_set(self) -> _VariousSizedImageSet:
        from safeds.data.image.containers import _VariousSizedImageSet
        if isinstance(self, _VariousSizedImageSet):
            return self
        raise ValueError("The given image_set is not a VariousSizedImageSet")

    def _as_fixed_sized_image_set(self) -> _FixedSizedImageSet:
        from safeds.data.image.containers import _FixedSizedImageSet
        if isinstance(self, _FixedSizedImageSet):
            return self
        raise ValueError("The given image_set is not a FixedSizedImageSet")

    # ------------------------------------------------------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def change_channel(self, channel: int) -> ImageSet:
        pass

    @abstractmethod
    def _add_image_tensor(self, image_tensor: Tensor, index: int) -> ImageSet:
        pass

    def add_image(self, image: Image) -> ImageSet:
        return self._add_image_tensor(image._image_tensor, self.number_of_images)

    def add_images(self, images: list[Image] | ImageSet) -> ImageSet:
        if isinstance(images, ImageSet):
            images = images.to_images()
        image_set = ImageSet.from_images(self.to_images(), self.indices)
        for image in images:
            image_set = image_set.add_image(image)
        return image_set

    def remove_image(self, image: Image) -> ImageSet:
        return self.remove_image_by_index(self.index(image))

    @abstractmethod
    def remove_image_by_index(self, index: int | list[int]) -> ImageSet:
        pass

    @abstractmethod
    def remove_images_with_size(self, width: int, height: int) -> ImageSet:
        pass

    def remove_duplicate_images(self) -> ImageSet:
        image_set = ImageSet.from_images(self.to_images(), self.indices)
        for image in self.to_images():
            image_set = image_set.remove_image_by_index(image_set.index(image)[1:])
        return image_set

    @abstractmethod
    def shuffle_images(self) -> ImageSet:
        pass

    @abstractmethod
    def resize(self, new_width: int, new_height: int) -> ImageSet:
        pass

    @abstractmethod
    def convert_to_grayscale(self) -> ImageSet:
        pass

    @abstractmethod
    def crop(self, x: int, y: int, width: int, height: int) -> ImageSet:
        pass

    @abstractmethod
    def flip_vertically(self) -> ImageSet:
        pass

    @abstractmethod
    def flip_horizontally(self) -> ImageSet:
        pass

    @abstractmethod
    def adjust_brightness(self, factor: float) -> ImageSet:
        pass

    @abstractmethod
    def add_noise(self, standard_deviation: float) -> ImageSet:
        pass

    @abstractmethod
    def adjust_contrast(self, factor: float) -> ImageSet:
        pass

    @abstractmethod
    def adjust_color_balance(self, factor: float) -> ImageSet:
        pass

    @abstractmethod
    def blur(self, radius: int) -> ImageSet:
        pass

    @abstractmethod
    def sharpen(self, factor: float) -> ImageSet:
        pass

    @abstractmethod
    def invert_colors(self) -> ImageSet:
        pass

    @abstractmethod
    def rotate_right(self) -> ImageSet:
        pass

    @abstractmethod
    def rotate_left(self) -> ImageSet:
        pass

    @abstractmethod
    def find_edges(self) -> ImageSet:
        pass
