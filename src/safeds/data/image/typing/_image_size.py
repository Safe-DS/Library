from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.exceptions import OutOfBoundsError, ClosedBound

if TYPE_CHECKING:
    from safeds.data.image.containers import Image


class ImageSize:
    """
    A container for image size data

    Parameters
    ----------
    width:
        the width of the image
    height:
        the height of the image
    channel:
        the channel of the image

    Raises
    ------
    OutOfBoundsError:
        if width or height are below 1
    ValueError
        if an invalid channel is given
    """

    def __init__(self, width: int, height: int, channel: int, *, _ignore_invalid_channel: bool = False) -> None:
        if width < 1 or height < 1:
            raise OutOfBoundsError(min(width, height), lower_bound=ClosedBound(1))
        elif not _ignore_invalid_channel and channel not in (1, 3, 4):
            raise ValueError(f"Channel {channel} is not a valid channel option. Use either 1, 3 or 4")
        elif channel < 1:
            raise OutOfBoundsError(channel, name="channel", lower_bound=ClosedBound(1))
        self._width = width
        self._height = height
        self._channel = channel

    @staticmethod
    def from_image(image: Image) -> ImageSize:
        return ImageSize(image.width, image.height, image.channel)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImageSize):
            return NotImplemented
        return self._width == other._width and self._height == other._height and self._channel == other._channel

    def __hash__(self):
        return _structural_hash(self._width, self._height, self._channel)

    def __sizeof__(self):
        return sys.getsizeof(self._width) + sys.getsizeof(self._height) + sys.getsizeof(self._channel)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def channel(self) -> int:
        return self._channel
