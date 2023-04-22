from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

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
