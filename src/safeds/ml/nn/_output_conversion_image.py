from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

from safeds.data.image.containers import ImageList
from safeds.data.labeled.containers import ImageDataset
from safeds.data.labeled.containers._image_dataset import _TableAsTensor, _ColumnAsTensor
from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
from safeds.data.tabular.containers import Table, Column
from safeds.data.tabular.transformation import OneHotEncoder

if TYPE_CHECKING:
    from torch import Tensor, LongTensor

from safeds.ml.nn._output_conversion import _OutputConversion

T = TypeVar("T", Column, Table, ImageList)


class _OutputConversionImage(_OutputConversion[ImageList, ImageDataset[T]], ABC):
    """The output conversion for a neural network, defines the output parameters for the neural network."""

    @abstractmethod
    def _data_conversion(self, **kwargs) -> ImageDataset[T]:
        pass


class OutputConversionImageToColumn(_OutputConversionImage[Column]):

    def _data_conversion(self, input_data: ImageList, output_data: Tensor, *, column_name: str, one_hot_encoder: OneHotEncoder) -> ImageDataset[Column]:
        import torch

        if not isinstance(input_data, _SingleSizeImageList):
            raise ValueError("The given input ImageList contains images of different sizes.")

        print(output_data)

        output = torch.zeros(len(input_data), len(one_hot_encoder.get_names_of_added_columns()))
        output[torch.arange(len(input_data)), output_data] = 1

        im_dataset: ImageDataset[Column] = ImageDataset[Column].__new__(ImageDataset)
        im_dataset._output = _ColumnAsTensor._from_tensor(output, column_name, one_hot_encoder)
        im_dataset._shuffle_tensor_indices = torch.LongTensor(list(range(len(input_data))))
        im_dataset._shuffle_after_epoch = False
        im_dataset._batch_size = 1
        im_dataset._next_batch_index = 0
        im_dataset._input_size = input_data.sizes[0]
        im_dataset._input = input_data
        return im_dataset


class OutputConversionImageToTable(_OutputConversionImage[Table]):

    def _data_conversion(self, input_data: ImageList, output_data: Tensor, *, column_names: list[str]) -> ImageDataset[Table]:
        import torch

        if not isinstance(input_data, _SingleSizeImageList):
            raise ValueError("The given input ImageList contains images of different sizes.")

        output = torch.zeros(len(input_data), len(column_names))
        output[torch.arange(len(input_data)), output_data] = 1

        im_dataset: ImageDataset[Table] = ImageDataset[Table].__new__(ImageDataset)
        im_dataset._output = _TableAsTensor._from_tensor(output, column_names)
        im_dataset._shuffle_tensor_indices = torch.LongTensor(list(range(len(input_data))))
        im_dataset._shuffle_after_epoch = False
        im_dataset._batch_size = 1
        im_dataset._next_batch_index = 0
        im_dataset._input_size = input_data.sizes[0]
        im_dataset._input = input_data
        return im_dataset


class OutputConversionImageToImage(_OutputConversionImage[ImageList]):

    def _data_conversion(self, input_data: ImageList, output_data: Tensor) -> ImageDataset[ImageList]:
        import torch

        if not isinstance(input_data, _SingleSizeImageList):
            raise ValueError("The given input ImageList contains images of different sizes.")

        return ImageDataset[ImageList](input_data, _SingleSizeImageList._create_from_tensor((output_data * 255).to(torch.uint8), list(
            range(output_data.size(dim=0)))))
