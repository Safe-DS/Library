from __future__ import annotations

import io
import sys
import warnings
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
    _check_resize_errors,
    _check_sharpen_errors_and_warnings,
)
from safeds.data.image.typing import ImageSize
from safeds.exceptions import IllegalFormatError

if TYPE_CHECKING:
    from torch import Tensor


class Image:
    """
    A container for image data.

    Parameters
    ----------
    image_tensor:
        The image data as tensor.
    """

    _filter_edges_kernel_cache: Tensor | None = None

    @staticmethod
    def _filter_edges_kernel() -> Tensor:
        import torch

        _init_default_device()

        if Image._filter_edges_kernel_cache is None:
            Image._filter_edges_kernel_cache = (
                torch.tensor([[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]], dtype=torch.float16)
                .unsqueeze(dim=0)
                .unsqueeze(dim=0)
                .to(_get_device())
            )
        if Image._filter_edges_kernel_cache.device != _get_device():
            Image._filter_edges_kernel_cache = Image._filter_edges_kernel_cache.to(_get_device())  # pragma: no cover

        return Image._filter_edges_kernel_cache

    @staticmethod
    def from_file(path: str | Path) -> Image:
        """
        Create an image from a file.

        Parameters
        ----------
        path:
            The path to the image file.

        Returns
        -------
        image:
            The image.

        Raises
        ------
        FileNotFoundError
            If the file of the path cannot be found
        """
        from torchvision.io import read_image

        if isinstance(path, str):
            path = Path(path)

        _init_default_device()

        if not path.is_file():
            raise FileNotFoundError(f"No such file or directory: '{path}'")

        return Image(image_tensor=read_image(str(path)).to(_get_device()))

    @staticmethod
    def from_bytes(data: bytes) -> Image:
        """
        Create an image from bytes.

        Parameters
        ----------
        data:
            The data of the image.

        Returns
        -------
        image:
            The image.
        """
        import torch
        import torchvision

        _init_default_device()

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given buffer is not writable, and PyTorch does not support non-writable tensors.",
            )
            input_tensor = torch.frombuffer(data, dtype=torch.uint8)

        return Image(image_tensor=torchvision.io.decode_image(input_tensor).to(_get_device()))

    def __init__(self, image_tensor: Tensor) -> None:
        import torch

        self._image_tensor: Tensor
        if image_tensor.dtype != torch.uint8:
            self._image_tensor = torch.clamp(image_tensor, 0, 255).to(torch.uint8)
        else:
            self._image_tensor = image_tensor

    def __eq__(self, other: object) -> bool:
        """
        Compare two images.

        Parameters
        ----------
        other:
            The image to compare to.

        Returns
        -------
        equals:
            Whether the two images contain equal pixel data.
        """
        import torch

        _init_default_device()

        if not isinstance(other, Image):
            return NotImplemented
        return (self is other) or (
            self._image_tensor.size() == other._image_tensor.size()
            and torch.all(torch.eq(self._image_tensor, other._image_tensor)).item()
        )

    def __hash__(self) -> int:
        """
        Return a deterministic hash value for this image.

        Returns
        -------
        hash:
            The hash value.
        """
        return _structural_hash(self.width, self.height, self.channel)

    def __sizeof__(self) -> int:
        """
        Return the complete size of this object.

        Returns
        -------
        size:
            Size of this object in bytes.
        """
        return sys.getsizeof(self._image_tensor) + self._image_tensor.element_size() * self._image_tensor.nelement()

    def _repr_jpeg_(self) -> bytes | None:
        """
        Return a JPEG image as bytes.

        If the image has an alpha channel return None.

        Returns
        -------
        jpeg:
            The image as JPEG.
        """
        import torch
        from torchvision.transforms.v2 import functional as func2
        from torchvision.utils import save_image

        _init_default_device()

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
        png:
            The image as PNG.
        """
        import torch
        from torchvision.transforms.v2 import functional as func2
        from torchvision.utils import save_image

        _init_default_device()

        buffer = io.BytesIO()
        if self.channel == 1:
            func2.to_pil_image(self._image_tensor, mode="L").save(buffer, format="png")
        else:
            save_image(self._image_tensor.to(torch.float32) / 255, buffer, format="png")
        buffer.seek(0)
        return buffer.read()

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def width(self) -> int:
        """
        Get the width of the image in pixels.

        Returns
        -------
        width:
            The width of the image.
        """
        return self._image_tensor.size(dim=2)

    @property
    def height(self) -> int:
        """
        Get the height of the image in pixels.

        Returns
        -------
        height:
            The height of the image.
        """
        return self._image_tensor.size(dim=1)

    @property
    def channel(self) -> int:
        """
        Get the number of channels of the image.

        Returns
        -------
        channel:
            The number of channels of the image.
        """
        return self._image_tensor.size(dim=0)

    @property
    def size(self) -> ImageSize:
        """
        Get the `ImageSize` of the image.

        Returns
        -------
        image_size:
            The size of the image.
        """
        return ImageSize(self.width, self.height, self.channel)

    # ------------------------------------------------------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------------------------------------------------------

    def to_jpeg_file(self, path: str | Path) -> None:
        """
        Save the image as a JPEG file.

        Parameters
        ----------
        path:
            The path to the JPEG file.
        """
        import torch
        from torchvision.transforms.v2 import functional as func2
        from torchvision.utils import save_image

        _init_default_device()

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
        path:
            The path to the PNG file.
        """
        import torch
        from torchvision.transforms.v2 import functional as func2
        from torchvision.utils import save_image

        _init_default_device()

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self.channel == 1:
            func2.to_pil_image(self._image_tensor, mode="L").save(path, format="png")
        else:
            save_image(self._image_tensor.to(torch.float32) / 255, path, format="png")

    # ------------------------------------------------------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------------------------------------------------------

    def change_channel(self, channel: int) -> Image:
        """
        Return a new `Image` that has the given number of channels.

        The original image is not modified.

        Parameters
        ----------
        channel:
            The new number of channels. 1 will result in a grayscale image.

        Returns
        -------
        result:
            The image with the given number of channels.

        Raises
        ------
        ValueError
            if the given channel is not a valid channel option
        """
        import torch

        _init_default_device()

        if self.channel == channel:
            image_tensor = self._image_tensor
        elif self.channel == 1 and channel == 3:
            image_tensor = torch.cat([self._image_tensor, self._image_tensor, self._image_tensor], dim=0)
        elif self.channel == 1 and channel == 4:
            image_tensor = torch.cat(
                [
                    self._image_tensor,
                    self._image_tensor,
                    self._image_tensor,
                    torch.full(self._image_tensor.size(), 255),
                ],
                dim=0,
            )
        elif self.channel in (3, 4) and channel == 1:
            image_tensor = self.convert_to_grayscale()._image_tensor[0:1]
        elif self.channel == 3 and channel == 4:
            image_tensor = torch.cat(
                [self._image_tensor, torch.full(self._image_tensor[0:1].size(), 255)],
                dim=0,
            )
        elif self.channel == 4 and channel == 3:
            image_tensor = self._image_tensor[0:3]
        else:
            raise ValueError(f"Channel {channel} is not a valid channel option. Use either 1, 3 or 4")
        return Image(image_tensor)

    def resize(self, new_width: int, new_height: int) -> Image:
        """
        Return a new `Image` that has been resized to a given size.

        The original image is not modified.

        Parameters
        ----------
        new_width:
            the new width of the image
        new_height:
            the new height of the image

        Returns
        -------
        result:
            The image with the given width and height.

        Raises
        ------
        OutOfBoundsError
            If new_width or new_height are below 1
        """
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2 import functional as func2

        _init_default_device()

        _check_resize_errors(new_width, new_height)
        return Image(
            func2.resize(self._image_tensor, size=[new_height, new_width], interpolation=InterpolationMode.NEAREST),
        )

    def convert_to_grayscale(self) -> Image:
        """
        Return a new `Image` that is converted to grayscale.

        The new image will have the same amount of channels as the original image.
        If you want to change the amount of channels used, please use the method [change_channel][safeds.data.image.containers._image.Image.change_channel].

        The original image is not modified.

        Returns
        -------
        result:
            The grayscale image.
        """
        import torch
        from torchvision.transforms.v2 import functional as func2

        _init_default_device()

        if self.channel == 1:
            return self
        elif self.channel == 4:
            return Image(
                torch.cat(
                    [
                        func2.rgb_to_grayscale(self._image_tensor[0:3], num_output_channels=3),
                        self._image_tensor[3:4],
                    ],
                ),
            )
        else:  # channel == 3
            return Image(
                func2.rgb_to_grayscale(self._image_tensor[0:3], num_output_channels=3),
            )

    def crop(self, x: int, y: int, width: int, height: int) -> Image:
        """
        Return a new `Image` that has been cropped to a given bounding rectangle.

        The original image is not modified.

        Parameters
        ----------
        x:
            the x coordinate of the top-left corner of the bounding rectangle
        y:
            the y coordinate of the top-left corner of the bounding rectangle
        width:
            the width of the bounding rectangle
        height:
            the height of the bounding rectangle

        Returns
        -------
        result:
            The cropped image.

        Raises
        ------
        OutOfBoundsError
            If x or y are below 0 or if width or height are below 1
        """
        from torchvision.transforms.v2 import functional as func2

        _init_default_device()

        _check_crop_errors(x, y, width, height)
        _check_crop_warnings(x, y, self.width, self.height, plural=False)
        return Image(func2.crop(self._image_tensor, y, x, height, width))

    def flip_top_and_bottom(self) -> Image:
        """
        Return a new `Image` where top and bottom are flipped along a horizontal axis.

        The original image is not modified.

        Returns
        -------
        result:
            The flipped image.
        """
        from torchvision.transforms.v2 import functional as func2

        _init_default_device()

        return Image(func2.vertical_flip(self._image_tensor))

    def flip_left_and_right(self) -> Image:
        """
        Return a new `Image` where left and right sides are flipped along a vertical axis.

        The original image is not modified.

        Returns
        -------
        result:
            The flipped image.
        """
        from torchvision.transforms.v2 import functional as func2

        _init_default_device()

        return Image(func2.horizontal_flip(self._image_tensor))

    def adjust_brightness(self, factor: float) -> Image:
        """
        Return a new `Image` with an adjusted brightness.

        The original image is not modified.

        Parameters
        ----------
        factor:
            The brightness factor.
            1.0 will not change the brightness.
            Below 1.0 will result in a darker image.
            Above 1.0 will resolut in a brighter image.
            Has to be bigger than or equal to 0 (black).

        Returns
        -------
        result:
            The Image with adjusted brightness.

        Raises
        ------
        OutOfBoundsError
            If factor is smaller than 0.
        """
        import torch

        _init_default_device()

        _check_adjust_brightness_errors_and_warnings(factor, plural=False)
        if self._image_tensor.size(dim=-3) != 4:
            if factor == 0:
                return Image(torch.zeros(self._image_tensor.size(), dtype=torch.uint8))
            else:
                temp_tensor = self._image_tensor * torch.tensor([factor * 1.0], dtype=torch.float16)
                torch.clamp(temp_tensor, 0, 255, out=temp_tensor)
                return Image(temp_tensor.to(torch.uint8))
        else:
            img_tensor = torch.empty(self._image_tensor.size(), dtype=torch.uint8)
            img_tensor[3] = self._image_tensor[3]
            if factor == 0:
                torch.zeros(
                    (3, self._image_tensor.size(dim=-2), self._image_tensor.size(dim=-1)),
                    dtype=torch.uint8,
                    out=img_tensor[:, 0:3],
                )
            else:
                temp_tensor = self._image_tensor[0:3] * torch.tensor([factor * 1.0], dtype=torch.float16)
                torch.clamp(temp_tensor, 0, 255, out=temp_tensor)
                img_tensor[0:3] = temp_tensor[:]
            return Image(img_tensor)

    def add_noise(self, standard_deviation: float) -> Image:
        """
        Return a new `Image` with noise added to the image.

        The original image is not modified.

        Parameters
        ----------
        standard_deviation:
            The standard deviation of the normal distribution. Has to be bigger than or equal to 0.

        Returns
        -------
        result:
            The image with added noise.

        Raises
        ------
        OutOfBoundsError
            If standard_deviation is smaller than 0.
        """
        import torch

        _init_default_device()

        _check_add_noise_errors(standard_deviation)
        float_tensor = torch.empty(self._image_tensor.size(), dtype=torch.float16)
        torch.normal(0, standard_deviation, self._image_tensor.size(), out=float_tensor)
        float_tensor *= 255
        float_tensor += self._image_tensor
        torch.clamp(float_tensor, 0, 255, out=float_tensor)
        return Image(float_tensor.to(torch.uint8))

    def adjust_contrast(self, factor: float) -> Image:
        """
        Return a new `Image` with adjusted contrast.

        The original image is not modified.

        Parameters
        ----------
        factor:
            If factor > 1, increase contrast of image.
            If factor = 1, no changes will be made.
            If factor < 1, make image greyer.
            Has to be bigger than or equal to 0 (gray).

        Returns
        -------
        image:
            New image with adjusted contrast.

        Raises
        ------
        OutOfBoundsError
            If factor is smaller than 0.
        """
        import torch

        _init_default_device()

        _check_adjust_contrast_errors_and_warnings(factor, plural=False)

        factor *= 1.0
        adjusted_factor = (1 - factor) / factor
        gray_tensor = self.convert_to_grayscale()._image_tensor[0]
        mean = torch.mean(gray_tensor, dim=(-2, -1), dtype=torch.float16)
        del gray_tensor
        mean *= torch.tensor(adjusted_factor, dtype=torch.float16)
        tensor = mean.repeat(min(self.channel, 3), self._image_tensor.size(dim=-2), self._image_tensor.size(dim=-1))
        tensor += self._image_tensor[0 : min(self.channel, 3)]
        tensor *= factor
        torch.clamp(tensor, 0, 255, out=tensor)

        if self.channel == 4:
            return Image(torch.cat([tensor.to(torch.uint8), self._image_tensor[3:4]], dim=0))
        else:
            return Image(tensor.to(torch.uint8))

    def adjust_color_balance(self, factor: float) -> Image:
        """
        Return a new `Image` with adjusted color balance.

        The original image is not modified.

        Parameters
        ----------
        factor:
            Has to be bigger than or equal to 0.
            If 0 <= factor < 1, make image greyer.
            If factor = 1, no changes will be made.
            If factor > 1, increase color balance of image.

        Returns
        -------
        image:
            The new, adjusted image.

        Raises
        ------
        OutOfBoundsError
            If factor is smaller than 0.
        """
        import torch

        _check_adjust_color_balance_errors_and_warnings(factor, self.channel, plural=False)

        factor *= 1.0
        if factor == 0:
            return self.convert_to_grayscale()
        else:
            adjusted_factor = (1 - factor) / factor
            tensor = self.convert_to_grayscale()._image_tensor * torch.tensor(adjusted_factor, dtype=torch.float16)
            tensor += self._image_tensor
            tensor *= factor
            torch.clamp(tensor, 0, 255, out=tensor)
            return Image(tensor.to(torch.uint8))

    def blur(self, radius: int) -> Image:
        """
        Return a blurred version of the image.

        The original image is not modified.

        Parameters
        ----------
        radius:
             Radius is directly proportional to the blur value. The radius is equal to the amount of pixels united in
             each direction. A radius of 1 will result in a united box of 9 pixels.

        Returns
        -------
        result:
            The blurred image.

        Raises
        ------
        OutOfBoundsError
            If radius is smaller than 0 or equal or greater than the smaller size of the image.
        """
        import torch

        _init_default_device()

        float_dtype = torch.float32 if _get_device() != torch.device("cuda") else torch.float16

        _check_blur_errors_and_warnings(radius, min(self.width, self.height), plural=False)

        kernel = torch.full(
            (self._image_tensor.size(dim=-3), 1, radius * 2 + 1, radius * 2 + 1),
            1 / (radius * 2 + 1) ** 2,
            dtype=float_dtype,
        )
        tensor = torch.nn.functional.conv2d(
            torch.nn.functional.pad(
                self._image_tensor.to(float_dtype),
                (radius, radius, radius, radius),
                mode="replicate",
            ),
            kernel,
            padding="valid",
            groups=self._image_tensor.size(dim=-3),
        ).to(torch.uint8)
        return Image(tensor)

    def sharpen(self, factor: float) -> Image:
        """
        Return a sharpened version of the image.

        The original image is not modified.

        Parameters
        ----------
        factor:
            If factor > 1, increase the sharpness of the image.
            If factor = 1, no changes will be made.
            If factor < 1, blur the image.
            Has to be bigger than or equal to 0 (blurred).

        Returns
        -------
        result:
            The image sharpened by the given factor.

        Raises
        ------
        OutOfBoundsError
            If factor is smaller than 0.
        """
        import torch
        from torchvision.transforms.v2 import functional as func2

        _init_default_device()

        _check_sharpen_errors_and_warnings(factor, plural=False)
        if self.channel == 4:
            return Image(
                torch.cat(
                    [
                        func2.adjust_sharpness(self._image_tensor[0:3], factor * 1.0),
                        self._image_tensor[3:4],
                    ],
                ),
            )
        else:
            return Image(func2.adjust_sharpness(self._image_tensor, factor * 1.0))

    def invert_colors(self) -> Image:
        """
        Return a new `Image` with colors inverted.

        The original image is not modified.

        Returns
        -------
        result:
            The image with inverted colors.
        """
        import torch
        from torchvision.transforms.v2 import functional as func2

        _init_default_device()

        if self.channel == 4:
            return Image(
                torch.cat([func2.invert(self._image_tensor[0:3]), self._image_tensor[3:4]]),
            )
        else:
            return Image(func2.invert(self._image_tensor))

    def rotate_right(self) -> Image:
        """
        Return a new `Image` that is rotated 90 degrees clockwise.

        The original image is not modified.

        Returns
        -------
        result:
            The image rotated 90 degrees clockwise.
        """
        from torchvision.transforms.v2 import functional as func2

        _init_default_device()

        return Image(func2.rotate(self._image_tensor, -90, expand=True))

    def rotate_left(self) -> Image:
        """
        Return a new `Image` that is rotated 90 degrees counter-clockwise.

        The original image is not modified.

        Returns
        -------
        result:
            The image rotated 90 degrees counter-clockwise.
        """
        from torchvision.transforms.v2 import functional as func2

        _init_default_device()

        return Image(func2.rotate(self._image_tensor, 90, expand=True))

    def find_edges(self) -> Image:
        """
        Return a grayscale version of the image with the edges highlighted.

        The original image is not modified.

        Returns
        -------
        result:
            The image with edges found.
        """
        import torch

        _init_default_device()

        edges_tensor_float16 = torch.nn.functional.conv2d(
            self.convert_to_grayscale()._image_tensor.to(torch.float16)[0:1],
            Image._filter_edges_kernel(),
            padding="same",
        )
        torch.clamp(edges_tensor_float16, 0, 255, out=edges_tensor_float16)
        if self.channel == 1:
            return Image(edges_tensor_float16.to(torch.uint8))
        edges_tensor = edges_tensor_float16.to(torch.uint8)
        del edges_tensor_float16
        if self.channel == 3:
            return Image(edges_tensor.repeat(3, 1, 1))
        else:  # self.channel == 4
            return Image(
                torch.cat([edges_tensor.repeat(3, 1, 1), self._image_tensor[3:4]]),
            )
