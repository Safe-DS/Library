from __future__ import annotations

import sys
import io
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as func
from PIL.Image import open as pil_image_open
from torch import Tensor

from safeds.config import _get_device

if TYPE_CHECKING:
    from torch.types import Device
import torchvision
from torchvision.transforms.v2 import PILToTensor
from torchvision.transforms.v2 import functional as func2
from torchvision.utils import save_image

from safeds.exceptions import ClosedBound, IllegalFormatError, OutOfBoundsError


class Image:
    """
    A container for image data.

    Parameters
    ----------
    image_tensor : Tensor
        The image data as tensor.
    """

    _pil_to_tensor = PILToTensor()
    _default_device = _get_device()
    _FILTER_EDGES_KERNEL = (
        torch.tensor([[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]])
        .unsqueeze(dim=0)
        .unsqueeze(dim=0)
        .to(_default_device)
    )

    @staticmethod
    def from_file(path: str | Path, device: Device = _default_device) -> Image:
        """
        Create an image from a file.

        Parameters
        ----------
        path : str | Path
            The path to the image file.
        device: Device
            The device where the tensor will be saved on. Defaults to the default device

        Returns
        -------
        image : Image
            The image.
        """
        return Image(image_tensor=Image._pil_to_tensor(pil_image_open(path)), device=device)

    @staticmethod
    def from_bytes(data: bytes, device: Device = _default_device) -> Image:
        """
        Create an image from bytes.

        Parameters
        ----------
        data : bytes
            The data of the image.
        device: Device
            The device where the tensor will be saved on. Defaults to the default device

        Returns
        -------
        image : Image
            The image.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given buffer is not writable, and PyTorch does not support non-writable tensors.",
            )
            input_tensor = torch.frombuffer(data, dtype=torch.uint8)
        return Image(image_tensor=torchvision.io.decode_image(input_tensor), device=device)

    def __init__(self, image_tensor: Tensor, device: Device = _default_device) -> None:
        self._image_tensor: Tensor = image_tensor.to(device)

    def __eq__(self, other: object) -> bool:
        """
        Compare two images.

        Parameters
        ----------
        other: The image to compare to.

        Returns
        -------
        equals : bool
            Whether the two images contain equal pixel data.
        """
        if not isinstance(other, Image):
            return NotImplemented
        return (
            self._image_tensor.size() == other._image_tensor.size()
            and torch.all(torch.eq(self._image_tensor, other._set_device(self.device)._image_tensor)).item()
        )

    def __sizeof__(self) -> int:
        """
        Return the complete size of this object.

        Returns
        -------
        Size of this object in bytes.
        """
        return sys.getsizeof(self._image_tensor) + self._image_tensor.element_size() * self._image_tensor.nelement()

    def _repr_jpeg_(self) -> bytes | None:
        """
        Return a JPEG image as bytes.

        If the image has an alpha channel return None.

        Returns
        -------
        jpeg : bytes
            The image as JPEG.
        """
        if self.channel == 4:
            return None
        buffer = io.BytesIO()
        if self.channel == 1:
            func2.to_pil_image(self._image_tensor, mode="L").save(buffer, format="jpeg")
        else:
            save_image(self._image_tensor.to(torch.float32) / 255, buffer, format="jpeg")
        buffer.seek(0)
        return buffer.read()

    def _repr_png_(self) -> bytes:
        """
        Return a PNG image as bytes.

        Returns
        -------
        png : bytes
            The image as PNG.
        """
        buffer = io.BytesIO()
        if self.channel == 1:
            func2.to_pil_image(self._image_tensor, mode="L").save(buffer, format="png")
        else:
            save_image(self._image_tensor.to(torch.float32) / 255, buffer, format="png")
        buffer.seek(0)
        return buffer.read()

    def _set_device(self, device: Device) -> Image:
        """
        Set the device where the image will be saved on.

        Returns
        -------
        result : Image
            The image on the given device
        """
        return Image(self._image_tensor, device)

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def width(self) -> int:
        """
        Get the width of the image in pixels.

        Returns
        -------
        width : int
            The width of the image.
        """
        return self._image_tensor.size(dim=2)

    @property
    def height(self) -> int:
        """
        Get the height of the image in pixels.

        Returns
        -------
        height : int
            The height of the image.
        """
        return self._image_tensor.size(dim=1)

    @property
    def channel(self) -> int:
        """
        Get the number of channels of the image.

        Returns
        -------
        channel : int
            The number of channels of the image.
        """
        return self._image_tensor.size(dim=0)

    @property
    def device(self) -> Device:
        """
        Get the device where the image is saved on.

        Returns
        -------
        device : Device
            The device of the image
        """
        return self._image_tensor.device

    # ------------------------------------------------------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------------------------------------------------------

    def to_jpeg_file(self, path: str | Path) -> None:
        """
        Save the image as a JPEG file.

        Parameters
        ----------
        path : str | Path
            The path to the JPEG file.
        """
        if self.channel == 4:
            raise IllegalFormatError("png")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self.channel == 1:
            func2.to_pil_image(self._image_tensor, mode="L").save(path, format="jpeg")
        else:
            save_image(self._image_tensor.to(torch.float32) / 255, path, format="jpeg")

    def to_png_file(self, path: str | Path) -> None:
        """
        Save the image as a PNG file.

        Parameters
        ----------
        path : str | Path
            The path to the PNG file.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self.channel == 1:
            func2.to_pil_image(self._image_tensor, mode="L").save(path, format="png")
        else:
            save_image(self._image_tensor.to(torch.float32) / 255, path, format="png")

    # ------------------------------------------------------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------------------------------------------------------

    def resize(self, new_width: int, new_height: int) -> Image:
        """
        Return a new `Image` that has been resized to a given size.

        The original image is not modified.

        Returns
        -------
        result : Image
            The image with the given width and height.
        """
        return Image(
            func.interpolate(self._image_tensor.unsqueeze(dim=1), size=(new_height, new_width)).squeeze(dim=1),
            device=self._image_tensor.device,
        )

    def convert_to_grayscale(self) -> Image:
        """
        Return a new `Image` that is converted to grayscale.

        The original image is not modified.

        Returns
        -------
        result : Image
            The grayscale image.
        """
        if self.channel == 4:
            return Image(
                torch.cat([
                    func2.rgb_to_grayscale(self._image_tensor[0:3], num_output_channels=3),
                    self._image_tensor[3].unsqueeze(dim=0),
                ]),
                device=self.device,
            )
        else:
            return Image(func2.rgb_to_grayscale(self._image_tensor[0:3], num_output_channels=3), device=self.device)

    def crop(self, x: int, y: int, width: int, height: int) -> Image:
        """
        Return a new `Image` that has been cropped to a given bounding rectangle.

        The original image is not modified.

        Parameters
        ----------
        x: the x coordinate of the top-left corner of the bounding rectangle
        y: the y coordinate of the top-left corner of the bounding rectangle
        width:  the width of the bounding rectangle
        height:  the height of the bounding rectangle

        Returns
        -------
        result : Image
            The cropped image.
        """
        return Image(func2.crop(self._image_tensor, x, y, height, width), device=self.device)

    def flip_vertically(self) -> Image:
        """
        Return a new `Image` that is flipped vertically (horizontal axis, flips up-down and vice versa).

        The original image is not modified.

        Returns
        -------
        result : Image
            The flipped image.
        """
        return Image(func2.vertical_flip(self._image_tensor), device=self.device)

    def flip_horizontally(self) -> Image:
        """
        Return a new `Ìmage` that is flipped horizontally (vertical axis, flips left-right and vice versa).

        The original image is not modified.

        Returns
        -------
        result : Image
            The flipped image.
        """
        return Image(func2.horizontal_flip(self._image_tensor), device=self.device)

    def adjust_brightness(self, factor: float) -> Image:
        """
        Return a new `Image` with an adjusted brightness.

        The original image is not modified.

        Parameters
        ----------
        factor: float
            The brightness factor.
            1.0 will not change the brightness.
            Below 1.0 will result in a darker image.
            Above 1.0 will resolut in a brighter image.
            Has to be bigger than or equal to 0 (black).

        Returns
        -------
        result: Image
            The Image with adjusted brightness.

        Raises
        ------
        OutOfBoundsError
            If factor is smaller than 0.
        """
        if factor < 0:
            raise OutOfBoundsError(factor, name="factor", lower_bound=ClosedBound(0))
        elif factor == 1:
            warnings.warn(
                "Brightness adjustment factor is 1.0, this will not make changes to the image.",
                UserWarning,
                stacklevel=2,
            )
        if self.channel == 4:
            return Image(
                torch.cat([
                    func2.adjust_brightness(self._image_tensor[0:3], factor * 1.0),
                    self._image_tensor[3].unsqueeze(dim=0),
                ]),
                device=self.device,
            )
        else:
            return Image(func2.adjust_brightness(self._image_tensor, factor * 1.0), device=self.device)

    def add_noise(self, standard_deviation: float) -> Image:
        """
        Return a new `Image` with noise added to the image.

        The original image is not modified.

        Parameters
        ----------
        standard_deviation : float
            The standard deviation of the normal distribution. Has to be bigger than or equal to 0.

        Returns
        -------
        result : Image
            The image with added noise.

        Raises
        ------
        OutOfBoundsError
            If standard_deviation is smaller than 0.
        """
        if standard_deviation < 0:
            raise OutOfBoundsError(standard_deviation, name="standard_deviation", lower_bound=ClosedBound(0))
        return Image(
            self._image_tensor + torch.normal(0, standard_deviation, self._image_tensor.size()).to(self.device) * 255,
            device=self.device,
        )

    def adjust_contrast(self, factor: float) -> Image:
        """
        Return a new `Image` with adjusted contrast.

        The original image is not modified.

        Parameters
        ----------
        factor: float
            If factor > 1, increase contrast of image.
            If factor = 1, no changes will be made.
            If factor < 1, make image greyer.
            Has to be bigger than or equal to 0 (gray).

        Returns
        -------
        image: Image
            New image with adjusted contrast.

        Raises
        ------
        OutOfBoundsError
            If factor is smaller than 0.
        """
        if factor < 0:
            raise OutOfBoundsError(factor, name="factor", lower_bound=ClosedBound(0))
        elif factor == 1:
            warnings.warn(
                "Contrast adjustment factor is 1.0, this will not make changes to the image.",
                UserWarning,
                stacklevel=2,
            )
        if self.channel == 4:
            return Image(
                torch.cat([
                    func2.adjust_contrast(self._image_tensor[0:3], factor * 1.0),
                    self._image_tensor[3].unsqueeze(dim=0),
                ]),
                device=self.device,
            )
        else:
            return Image(func2.adjust_contrast(self._image_tensor, factor * 1.0), device=self.device)

    def adjust_color_balance(self, factor: float) -> Image:
        """
        Return a new `Image` with adjusted color balance.

        The original image is not modified.

        Parameters
        ----------
        factor: float
            Has to be bigger than or equal to 0.
            If 0 <= factor < 1, make image greyer.
            If factor = 1, no changes will be made.
            If factor > 1, increase color balance of image.

        Returns
        -------
        image: Image
            The new, adjusted image.
        """
        if factor < 0:
            raise OutOfBoundsError(factor, name="factor", lower_bound=ClosedBound(0))
        elif factor == 1:
            warnings.warn(
                "Color adjustment factor is 1.0, this will not make changes to the image.",
                UserWarning,
                stacklevel=2,
            )
        elif self.channel == 1:
            warnings.warn(
                "Color adjustment will not have an affect on grayscale images with only one channel.",
                UserWarning,
                stacklevel=2,
            )
        return Image(
            self.convert_to_grayscale()._image_tensor * (1.0 - factor * 1.0) + self._image_tensor * (factor * 1.0),
            device=self.device,
        )

    def blur(self, radius: int) -> Image:
        """
        Return a blurred version of the image.

        The original image is not modified.

        Parameters
        ----------
        radius : int
             Radius is directly proportional to the blur value. The radius is equal to the amount of pixels united in
             each direction. A radius of 1 will result in a united box of 9 pixels.

        Returns
        -------
        result : Image
            The blurred image.
        """
        return Image(func2.gaussian_blur(self._image_tensor, [radius * 2 + 1, radius * 2 + 1]), device=self.device)

    def sharpen(self, factor: float) -> Image:
        """
        Return a sharpened version of the image.

        The original image is not modified.

        Parameters
        ----------
        factor : float
            If factor > 1, increase the sharpness of the image.
            If factor = 1, no changes will be made.
            If factor < 1, blur the image.
            Has to be bigger than or equal to 0 (blurred).

        Returns
        -------
        result : Image
            The image sharpened by the given factor.

        Raises
        ------
        OutOfBoundsError
            If factor is smaller than 0.
        """
        if factor < 0:
            raise OutOfBoundsError(factor, name="factor", lower_bound=ClosedBound(0))
        elif factor == 1:
            warnings.warn(
                "Sharpen factor is 1.0, this will not make changes to the image.",
                UserWarning,
                stacklevel=2,
            )
        if self.channel == 4:
            return Image(
                torch.cat([
                    func2.adjust_sharpness(self._image_tensor[0:3], factor * 1.0),
                    self._image_tensor[3].unsqueeze(dim=0),
                ]),
                device=self.device,
            )
        else:
            return Image(func2.adjust_sharpness(self._image_tensor, factor * 1.0), device=self.device)

    def invert_colors(self) -> Image:
        """
        Return a new `Image` with colors inverted.

        The original image is not modified.

        Returns
        -------
        result : Image
            The image with inverted colors.
        """
        if self.channel == 4:
            return Image(
                torch.cat([func2.invert(self._image_tensor[0:3]), self._image_tensor[3].unsqueeze(dim=0)]),
                device=self.device,
            )
        else:
            return Image(func2.invert(self._image_tensor), device=self.device)

    def rotate_right(self) -> Image:
        """
        Return a new `Image` that is rotated 90 degrees clockwise.

        The original image is not modified.

        Returns
        -------
        result : Image
            The image rotated 90 degrees clockwise.
        """
        return Image(func2.rotate(self._image_tensor, -90, expand=True), device=self.device)

    def rotate_left(self) -> Image:
        """
        Return a new `Image` that is rotated 90 degrees counter-clockwise.

        The original image is not modified.

        Returns
        -------
        result : Image
            The image rotated 90 degrees counter-clockwise.
        """
        return Image(func2.rotate(self._image_tensor, 90, expand=True), device=self.device)

    def find_edges(self) -> Image:
        """
        Return a grayscale version of the image with the edges highlighted.

        The original image is not modified.

        Returns
        -------
        result : Image
            The image with edges found.
        """
        kernel = (
            Image._FILTER_EDGES_KERNEL
            if self.device.type == Image._default_device
            else Image._FILTER_EDGES_KERNEL.to(self.device)
        )
        edges_tensor = torch.clamp(
            torch.nn.functional.conv2d(
                self.convert_to_grayscale()._image_tensor.float()[0].unsqueeze(dim=0),
                kernel,
                padding="same",
            ).squeeze(dim=1),
            0,
            255,
        ).to(torch.uint8)
        if self.channel == 3:
            return Image(edges_tensor.repeat(3, 1, 1), device=self.device)
        elif self.channel == 4:
            return Image(
                torch.cat([edges_tensor.repeat(3, 1, 1), self._image_tensor[3].unsqueeze(dim=0)]),
                device=self.device,
            )
        else:
            return Image(edges_tensor, device=self.device)
