from __future__ import annotations

import copy
import io
import warnings
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import PIL
from PIL import ImageEnhance, ImageFilter, ImageOps
from PIL.Image import Image as PillowImage
from PIL.Image import open as open_image
from skimage.util import random_noise

from safeds.data.image.typing import ImageFormat
from safeds.exceptions import ClosedBound, OutOfBoundsError


class Image:
    """
    A container for image data.

    Parameters
    ----------
    data : BinaryIO
        The image data as bytes.
    """

    @staticmethod
    def from_jpeg_file(path: str | Path) -> Image:
        """
        Create an image from a JPEG file.

        Parameters
        ----------
        path : str | Path
            The path to the JPEG file.

        Returns
        -------
        image : Image
            The image.
        """
        return Image(
            data=Path(path).open("rb"),
            format_=ImageFormat.JPEG,
        )

    @staticmethod
    def from_png_file(path: str | Path) -> Image:
        """
        Create an image from a PNG file.

        Parameters
        ----------
        path : str | Path
            The path to the PNG file.

        Returns
        -------
        image : Image
            The image.
        """
        return Image(
            data=Path(path).open("rb"),
            format_=ImageFormat.PNG,
        )

    def __init__(self, data: BinaryIO, format_: ImageFormat):
        data.seek(0)

        self._image: PillowImage = open_image(data, formats=[format_.value])
        self._format: ImageFormat = format_

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def format(self) -> ImageFormat:
        """
        Get the image format.

        Returns
        -------
        format : ImageFormat
            The image format.
        """
        return self._format

    @property
    def width(self) -> int:
        """
        Get the width of the image in pixels.

        Returns
        -------
        width : int
            The width of the image.
        """
        return self._image.width

    @property
    def height(self) -> int:
        """
        Get the height of the image in pixels.

        Returns
        -------
        height : int
            The height of the image.
        """
        return self._image.height

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
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._image.save(path, format="jpeg")

    def to_png_file(self, path: str | Path) -> None:
        """
        Save the image as a PNG file.

        Parameters
        ----------
        path : str | Path
            The path to the PNG file.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._image.save(path, format="png")

    # ------------------------------------------------------------------------------------------------------------------
    # IPython integration
    # ------------------------------------------------------------------------------------------------------------------

    def __eq__(self, other: Any) -> bool:
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
        return self._image.tobytes() == other._image.tobytes()

    def _repr_jpeg_(self) -> bytes | None:
        """
        Return a JPEG image as bytes.

        If the image is not a JPEG, return None.

        Returns
        -------
        jpeg : bytes | None
            The image as JPEG.
        """
        if self._format != ImageFormat.JPEG:
            return None

        buffer = io.BytesIO()
        self._image.save(buffer, format="jpeg")
        buffer.seek(0)
        return buffer.read()

    def _repr_png_(self) -> bytes | None:
        """
        Return a PNG image as bytes.

        If the image is not a PNG, return None.

        Returns
        -------
        png : bytes | None
            The image as PNG.
        """
        if self._format != ImageFormat.PNG:
            return None

        buffer = io.BytesIO()
        self._image.save(buffer, format="png")
        buffer.seek(0)
        return buffer.read()

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
        image_copy = copy.deepcopy(self)
        image_copy._image = image_copy._image.resize((new_width, new_height))
        return image_copy

    def convert_to_grayscale(self) -> Image:
        """
        Return a new `Image` that is converted to grayscale.

        The original image is not modified.

        Returns
        -------
        result : Image
            The grayscale image.
        """
        image_copy = copy.deepcopy(self)
        image_copy._image = image_copy._image.convert("L")
        return image_copy

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
        image_copy = copy.deepcopy(self)
        image_copy._image = image_copy._image.crop((x, y, (x + width), (y + height)))
        return image_copy

    def flip_vertically(self) -> Image:
        """
        Return a new `Image` that is flipped vertically (horizontal axis, flips up-down and vice versa).

        The original image is not modified.

        Returns
        -------
        result : Image
            The flipped image.
        """
        image_copy = copy.deepcopy(self)
        image_copy._image = self._image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        return image_copy

    def flip_horizontally(self) -> Image:
        """
        Return a new `Ìmage` that is flipped horizontally (vertical axis, flips left-right and vice versa).

        The original image is not modified.

        Returns
        -------
        result : Image
            The flipped image.
        """
        image_copy = copy.deepcopy(self)
        image_copy._image = self._image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        return image_copy

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
        """
        if factor < 0:
            raise OutOfBoundsError(factor, name="factor", lower_bound=ClosedBound(0))
        elif factor == 1:
            warnings.warn(
                "Brightness adjustment factor is 1.0, this will not make changes to the image.",
                UserWarning,
                stacklevel=2,
            )

        image_copy = copy.deepcopy(self)
        image_copy._image = ImageEnhance.Brightness(image_copy._image).enhance(factor)
        return image_copy

    def add_gaussian_noise(self, standard_deviation: float) -> Image:
        """
        Return a new `Image` with Gaussian noise added to the image.

        The original image is not modified.

        Parameters
        ----------
        standard_deviation : float
            The standard deviation of the Gaussian distribution. Has to be bigger than or equal to 0.

        Returns
        -------
        result : Image
            The image with added Gaussian noise.

        Raises
        ------
        OutOfBoundsError
            If standard_deviation is smaller than 0.
        """
        if standard_deviation < 0:
            raise OutOfBoundsError(standard_deviation, name="standard_deviation", lower_bound=ClosedBound(0))

        # noinspection PyTypeChecker
        image_as_array = np.asarray(self._image)
        noisy_image_as_array = random_noise(
            image_as_array,
            mode="gaussian",
            var=standard_deviation**2,
            rng=42,
            clip=True,
        )
        noisy_image = PIL.Image.fromarray(np.uint8(255 * noisy_image_as_array))

        image_copy = copy.deepcopy(self)
        image_copy._image = noisy_image
        return image_copy

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
        """
        if factor < 0:
            raise OutOfBoundsError(factor, name="factor", lower_bound=ClosedBound(0))
        elif factor == 1:
            warnings.warn(
                "Contrast adjustment factor is 1.0, this will not make changes to the image.",
                UserWarning,
                stacklevel=2,
            )

        image_copy = copy.deepcopy(self)
        image_copy._image = ImageEnhance.Contrast(image_copy._image).enhance(factor)
        return image_copy

    def adjust_color_balance(self, factor: float) -> Image:
        """
        Return a new `Image` with adjusted color balance.

        The original image is not modified.

        Parameters
        ----------
        factor: float
            If factor > 1, increase color balance of image.
            If factor = 1, no changes will be made.
            If factor < 1, make image greyer.
            Has to be bigger than or equal to 0.

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

        image_copy = copy.deepcopy(self)
        image_copy._image = ImageEnhance.Color(image_copy._image).enhance(factor)
        return image_copy

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
        image_copy = copy.deepcopy(self)
        image_copy._image = image_copy._image.filter(ImageFilter.BoxBlur(radius))
        return image_copy

    def sharpen(self, factor: float) -> Image:
        """
        Return a sharpened version of the image.

        The original image is not modified.

        Parameters
        ----------
        factor : float
            The amount of sharpness to be applied to the image. Factor 1.0 is considered to be neutral and does not make
            any changes.

        Returns
        -------
        result : Image
            The image sharpened by the given factor.
        """
        image_copy = copy.deepcopy(self)
        image_copy._image = ImageEnhance.Sharpness(image_copy._image).enhance(factor)
        return image_copy

    def invert_colors(self) -> Image:
        """
        Return a new image with colors inverted.

        The original image is not modified.

        Returns
        -------
        result : Image
            The image with inverted colors.
        """
        image_copy = copy.deepcopy(self)
        image_copy._image = ImageOps.invert(image_copy._image.convert("RGB"))
        return image_copy

    def rotate_right(self) -> Image:
        """
        Return a new `Image` that is rotated 90 degrees clockwise.

        The original image is not modified.

        Returns
        -------
        result : Image
            The image rotated 90 degrees clockwise.
        """
        image_copy = copy.deepcopy(self)
        image_copy._image = image_copy._image.rotate(270, expand=True)
        return image_copy

    def rotate_left(self) -> Image:
        """
        Return a new `Image` that is rotated 90 degrees counter-clockwise.

        The original image is not modified.

        Returns
        -------
        result : Image
            The image rotated 90 degrees counter-clockwise.
        """
        image_copy = copy.deepcopy(self)
        image_copy._image = image_copy._image.rotate(90, expand=True)
        return image_copy

    def find_edges(self) -> Image:
        """
        Return a grayscale version of the image with the edges highlighted.

        The original image is not modified.

        Returns
        -------
        result : Image
            The image with edges found.
        """
        image_copy = copy.deepcopy(self)
        image_copy = image_copy.convert_to_grayscale()
        image_copy._image = image_copy._image.filter(ImageFilter.FIND_EDGES)
        return image_copy
