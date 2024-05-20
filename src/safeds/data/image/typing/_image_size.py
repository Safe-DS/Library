from safeds.ml.nn.typing import ConstantImageSize


class ImageSize(ConstantImageSize):
    """
    A container for image size data.

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

    def __str__(self) -> str:
        return f"{self._width}x{self._height}x{self._channel} (WxHxC)"
