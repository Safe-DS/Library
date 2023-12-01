from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL.Image import open as pil_image_open
from torch import Tensor
from torch.types import Device
from torchvision.transforms.v2 import PILToTensor, functional as F2
from torchvision.utils import save_image

from safeds.exceptions import OutOfBoundsError, ClosedBound


class Image:
    _pil_to_tensor = PILToTensor()

    @staticmethod
    def from_file(path: str | Path, device: Device):
        return Image(image_tensor=Image._pil_to_tensor(pil_image_open(path)), device=device)

    def __init__(self, image_tensor: Tensor, device: Device) -> None:
        self._image_tensor: Tensor = image_tensor.to(device)
        # self._device = device

    def _repr_jpeg_(self) -> bytes | None:
        buffer = io.BytesIO()
        save_image(self._image_tensor.to(torch.float32) / 255, buffer, format="jpeg")
        buffer.seek(0)
        return buffer.read()

    def _repr_png_(self) -> bytes | None:
        buffer = io.BytesIO()
        save_image(self._image_tensor.to(torch.float32) / 255, buffer, format="png")
        buffer.seek(0)
        return buffer.read()

    @property
    def width(self) -> int:
        return self._image_tensor.size(dim=2)

    @property
    def height(self) -> int:
        return self._image_tensor.size(dim=1)

    @property
    def device(self) -> Device:
        return self._image_tensor.device

    def set_device(self, device: Device) -> Image:
        return Image(self._image_tensor, device)

    # def to_file(self, path: str | Path, file_format: str):
    #     torchvision.utils.save_image(self._image_tensor, path, file_format)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Image):
            return NotImplemented
        return self._image_tensor.size() == other._image_tensor.size() and torch.all(
            torch.eq(self._image_tensor, other.set_device(self.device)._image_tensor)).item()

    def resize(self, new_width: int, new_height: int) -> Image:
        return Image(F.interpolate(self._image_tensor.unsqueeze(dim=1), size=(new_height, new_width)).squeeze(dim=1),
                     device=self._image_tensor.device)

    def convert_to_grayscale(self) -> Image:
        return Image(F2.rgb_to_grayscale(self._image_tensor[0:3], num_output_channels=3), device=self.device)

    def crop(self, x: int, y: int, width: int, height: int) -> Image:
        return Image(F2.crop(self._image_tensor, x, y, height, width), device=self.device)

    def flip_vertically(self) -> Image:
        return Image(F2.vertical_flip(self._image_tensor), device=self.device)

    def flip_horizontally(self) -> Image:
        return Image(F2.horizontal_flip(self._image_tensor), device=self.device)

    def adjust_brightness(self, factor: float) -> Image:
        print(factor)
        if factor < 0:
            raise OutOfBoundsError(factor, name="factor", lower_bound=ClosedBound(0))
        elif factor == 1:
            warnings.warn(
                "Brightness adjustment factor is 1.0, this will not make changes to the image.",
                UserWarning,
                stacklevel=2,
            )
        if self._image_tensor.size(dim=0) == 4:
            return Image(torch.cat(
                [F2.adjust_brightness(self._image_tensor[0:3], factor * 1.0), self._image_tensor[3].unsqueeze(dim=0)]),
                device=self.device)
        else:
            return Image(F2.adjust_brightness(self._image_tensor, factor * 1.0), device=self.device)

    def add_gaussian_noise(self, standard_deviation: float) -> Image:
        # TODO: Different noise
        if standard_deviation < 0:
            raise OutOfBoundsError(standard_deviation, name="standard_deviation", lower_bound=ClosedBound(0))
        return Image(
            self._image_tensor + torch.normal(0, standard_deviation, self._image_tensor.size()).to(self.device) * 255,
            device=self.device)

    def adjust_contrast(self, factor: float) -> Image:
        if factor < 0:
            raise OutOfBoundsError(factor, name="factor", lower_bound=ClosedBound(0))
        elif factor == 1:
            warnings.warn(
                "Contrast adjustment factor is 1.0, this will not make changes to the image.",
                UserWarning,
                stacklevel=2,
            )
        if self._image_tensor.size(dim=0) == 4:
            return Image(torch.cat(
                [F2.adjust_contrast(self._image_tensor[0:3], factor * 1.0), self._image_tensor[3].unsqueeze(dim=0)]),
                device=self.device)
        else:
            return Image(F2.adjust_contrast(self._image_tensor, factor * 1.0), device=self.device)

    def adjust_color_balance(self, factor: float) -> Image:
        pass

    def blur(self, radius: int) -> Image:
        # TODO: Different blur
        return Image(F2.gaussian_blur(self._image_tensor, [radius * 2 + 1, radius * 2 + 1]), device=self.device)

    def sharpen(self, factor: float) -> Image:
        if factor < 0:
            raise OutOfBoundsError(factor, name="factor", lower_bound=ClosedBound(0))
        elif factor == 1:
            warnings.warn(
                "Sharpen factor is 1.0, this will not make changes to the image.",
                UserWarning,
                stacklevel=2,
            )
        if self._image_tensor.size(dim=0) == 4:
            return Image(torch.cat(
                [F2.adjust_sharpness(self._image_tensor[0:3], factor * 1.0), self._image_tensor[3].unsqueeze(dim=0)]),
                device=self.device)
        else:
            return Image(F2.adjust_sharpness(self._image_tensor, factor * 1.0), device=self.device)

    def invert_colors(self) -> Image:
        if self._image_tensor.size(dim=0) == 4:
            return Image(torch.cat(
                [F2.invert(self._image_tensor[0:3]), self._image_tensor[3].unsqueeze(dim=0)]),
                device=self.device)
        else:
            return Image(F2.invert(self._image_tensor), device=self.device)

    def rotate_right(self) -> Image:
        return Image(F2.rotate(self._image_tensor, -90, expand=True), device=self.device)

    def rotate_left(self) -> Image:
        return Image(F2.rotate(self._image_tensor, 90, expand=True), device=self.device)

    def find_edges(self) -> Image:
        pass
