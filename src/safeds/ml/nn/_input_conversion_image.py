from __future__ import annotations

from safeds.data.image.containers import ImageDataset, ImageList
from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
from safeds.data.image.typing import ImageSize

from safeds.ml.nn._input_conversion import _InputConversion


class InputConversionImage(_InputConversion[ImageDataset, ImageList]):
    """The input conversion for a neural network, defines the input parameters for the neural network."""

    def __init__(self, image_size: ImageSize) -> None:
        """
        Define the input parameters for the neural network in the input conversion.

        Parameters
        ----------
        """
        self._image_size = image_size

    @property
    def _data_size(self) -> ImageSize:
        return self._image_size

    def _data_conversion_fit(self, input_data: ImageDataset, batch_size: int, num_of_classes: int = 1) -> ImageDataset:
        return input_data

    def _data_conversion_predict(self, input_data: ImageList, batch_size: int) -> ImageList:
        return input_data

    def _is_fit_data_valid(self, input_data: ImageDataset) -> bool:
        return input_data.input_size == self._image_size

    def _is_predict_data_valid(self, input_data: ImageList) -> bool:
        return isinstance(input_data, _SingleSizeImageList) and input_data.sizes[0] == self._image_size
