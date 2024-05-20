from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self

from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _ClosedBound

if TYPE_CHECKING:
    from safeds.data.image.containers import Image


class ModelImageSize(ABC):
    """
    A container for image size in neural networks.

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
        _check_bounds("width", width, lower_bound=_ClosedBound(1))
        _check_bounds("height", height, lower_bound=_ClosedBound(1))
        if not _ignore_invalid_channel and channel not in (1, 3, 4):
            raise ValueError(f"Channel {channel} is not a valid channel option. Use either 1, 3 or 4")
        _check_bounds("channel", channel, lower_bound=_ClosedBound(1))

        self._width = width
        self._height = height
        self._channel = channel

    @classmethod
    def from_image(cls: type[Self], image: Image) -> Self:
        """
        Create a `ModelImageSize` of a given image.

        Parameters
        ----------
        image:
            the given image for the `ModelImageSize`

        Returns
        -------
        image_size:
            the calculated `ModelImageSize`
        """
        return cls(image.width, image.height, image.channel)

    @classmethod
    def from_image_size(cls: type[Self], image_size: ModelImageSize) -> Self:
        """
        Create a `ModelImageSize` of a given image size.

        Parameters
        ----------
        image_size:
            the given image size for the `ModelImageSize`

        Returns
        -------
        image_size:
            the new `ModelImageSize`
        """
        return cls(image_size.width, image_size.height, image_size.channel)

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelImageSize):
            return NotImplemented
        return (self is other) or (
            self._width == other._width and self._height == other._height and self._channel == other._channel
        )

    @abstractmethod
    def __hash__(self) -> int:
        return _structural_hash(self.__class__.__name__, self._width, self._height, self._channel)

    def __sizeof__(self) -> int:
        return sys.getsizeof(self._width) + sys.getsizeof(self._height) + sys.getsizeof(self._channel)

    def __str__(self) -> str:
        return f"{self._width}x{self._height}x{self._channel} (WxHxC)"

    @property
    def width(self) -> int:
        """
        Get the width of this `ImageSize` in pixels.

        Returns
        -------
        width:
            The width of this `ImageSize`.
        """
        return self._width

    @property
    def height(self) -> int:
        """
        Get the height of this `ImageSize` in pixels.

        Returns
        -------
        height:
            The height of this `ImageSize`.
        """
        return self._height

    @property
    def channel(self) -> int:
        """
        Get the channel of this `ImageSize` in pixels.

        Returns
        -------
        channel:
            The channel of this `ImageSize`.
        """
        return self._channel


class ConstantImageSize(ModelImageSize):
    """
    A container for constant image size in neural networks.

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

    def __eq__(self, other: object) -> bool:
        if isinstance(other, VariableImageSize):
            return other.__eq__(self)
        else:
            return super().__eq__(other)

    def __hash__(self) -> int:
        return super().__hash__()

    def __str__(self) -> str:
        return f"ConstantImageSize | {super().__str__()}"


class VariableImageSize(ModelImageSize):
    """
    A container for variable image size in neural networks.

    With a `VariableImageSize`, all image sizes that are a multiple of `width` and `height` are valid.

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelImageSize):
            return NotImplemented
        return (self is other) or (
            not (self._width % other._width and other._width % self._width)
            and not (self._height % other._height and other._height % self._height)
            and self._channel == other._channel
        )

    def __hash__(self) -> int:
        return super().__hash__()

    def __str__(self) -> str:
        return f"VariableImageSize | {super().__str__()}"
