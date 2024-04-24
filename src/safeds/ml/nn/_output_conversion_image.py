from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

from safeds.data.image.containers import ImageDataset, ImageList
from safeds.data.image.containers._image_dataset import _TableAsTensor
from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
from safeds.data.tabular.containers import Table

if TYPE_CHECKING:
    from torch import Tensor, LongTensor

from safeds.ml.nn._output_conversion import _OutputConversion

T = TypeVar("T", Table, ImageList)


class _OutputConversionImage(_OutputConversion[ImageList, ImageDataset[T]], ABC):
    """The output conversion for a neural network, defines the output parameters for the neural network."""

    @abstractmethod
    def _data_conversion(self, input_data: ImageList, output_data: Tensor) -> ImageDataset[T]:
        pass


class OutputConversionImageToTable(_OutputConversionImage[Table]):

    def _data_conversion(self, input_data: ImageList, output_data: Tensor) -> ImageDataset[Table]:
        from torch import LongTensor

        if not isinstance(input_data, _SingleSizeImageList):
            raise ValueError("The given input ImageList contains images of different sizes.")

        im_dataset = ImageDataset.__new__(ImageDataset)
        im_dataset._output = _TableAsTensor._from_tensor(output_data)
        im_dataset._shuffle_tensor_indices = LongTensor(list(range(len(input_data))))
        im_dataset._shuffle_after_epoch = False
        im_dataset._batch_size = 1
        im_dataset._next_batch_index = 0
        im_dataset._input_size = input_data.sizes[0]
        im_dataset._input = input_data
        return im_dataset


class OutputConversionImageToImage(_OutputConversionImage[ImageList]):

    def _data_conversion(self, input_data: ImageList, output_data: Tensor) -> ImageDataset[ImageList]:
        if not isinstance(input_data, _SingleSizeImageList):
            raise ValueError("The given input ImageList contains images of different sizes.")

        return ImageDataset(input_data, _SingleSizeImageList._create_from_tensor(output_data, list(
            range(output_data.size(dim=0)))))
