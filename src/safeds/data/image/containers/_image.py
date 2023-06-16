from __future__ import annotations

import copy
import io
from pathlib import Path
from typing import Any, BinaryIO

import PIL
from PIL import ImageFilter
from PIL.Image import Image as PillowImage
from PIL.Image import open as open_image

from safeds.data.image.typing import ImageFormat


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
        jpeg : bytes
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
        png : bytes
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
        Return an image that has been resized to a given size.

        Returns
        -------
        result : Image
            The image with the given width and height
        """
        data = io.BytesIO()
        repr_png = self._repr_png_()
        repr_jpeg = self._repr_jpeg_()
        if repr_png is not None:
            data = io.BytesIO(repr_png)
        elif repr_jpeg is not None:
            data = io.BytesIO(repr_jpeg)

        new_image = Image(data, self._format)
        new_image._image = new_image._image.resize((new_width, new_height))
        return new_image

    def convert_to_grayscale(self) -> Image:
        """
        Convert the image to grayscale.

        Returns
        -------
        grayscale_image : Image
            The grayscale image.
        """
        data = io.BytesIO()
        grayscale_image = self._image.convert("L")
        grayscale_image.save(data, format=self._format.value)
        return Image(data, self._format)

    def crop(self, x: int, y: int, width: int, height: int) -> Image:
        """
        Return an image that has been cropped to a given bounding rectangle.

        Parameters
        ----------
        x: the x coordinate of the top-left corner of the bounding rectangle
        y: the y coordinate of the top-left corner of the bounding rectangle
        width:  the width of the bounding rectangle
        height:  the height of the bounding rectangle

        Returns
        -------
        result : Image
            The image with the
        """
        data = io.BytesIO()
        repr_png = self._repr_png_()
        repr_jpeg = self._repr_jpeg_()
        if repr_png is not None:
            data = io.BytesIO(repr_png)
        elif repr_jpeg is not None:
            data = io.BytesIO(repr_jpeg)

        new_image = Image(data, self._format)
        new_image._image = new_image._image.crop((x, y, (x + width), (y + height)))
        return new_image

    def flip_vertically(self) -> Image:
        """
        Flip the image vertically (horizontal axis, flips up-down and vice versa).

        Returns
        -------
        result : Image
            The flipped image.
        """
        imagecopy = copy.deepcopy(self)
        imagecopy._image = self._image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        return imagecopy

    def flip_horizontally(self) -> Image:
        """
        Flip the image horizontally (vertical axis, flips left-right and vice versa).

        Returns
        -------
        result : Image
            The flipped image.
        """
        imagecopy = copy.deepcopy(self)
        imagecopy._image = self._image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        return imagecopy

    def blur(self, radius: int = 1) -> Image:
        """
        Return the blurred image.

        Parameters
        ----------
        radius : int
             Radius is directly proportional to the blur value. The radius is equal to the amount of pixels united in each direction.
             A Radius of 1 will result in a united box of 9 pixels.

        Returns
        -------
        result : Image
            The blurred image
        """
        data = io.BytesIO()
        repr_png = self._repr_png_()
        repr_jpeg = self._repr_jpeg_()
        if repr_png is not None:
            data = io.BytesIO(repr_png)
        elif repr_jpeg is not None:
            data = io.BytesIO(repr_jpeg)

        new_image = Image(data, self._format)
        new_image._image = new_image._image.filter(ImageFilter.BoxBlur(radius))
        return new_image
