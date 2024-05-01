from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

from safeds.data.image.containers import ImageList
from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
from safeds.data.labeled.containers import ImageDataset
from safeds.data.labeled.containers._image_dataset import _TableAsTensor, _ColumnAsTensor
from safeds.data.tabular.containers import Table, Column

if TYPE_CHECKING:
    from torch import Tensor, LongTensor

from safeds.ml.nn._output_conversion import _OutputConversion

from safeds.data.tabular.transformation import OneHotEncoder


class _OutputConversionImage:
    pass  # pragma: no cover


class OutputConversionImageToColumn(_OutputConversion[ImageList, ImageDataset[Column]], _OutputConversionImage):

    def _data_conversion(self, input_data: ImageList, output_data: Tensor, **kwargs: Any) -> ImageDataset[Column]:
        import torch

        if not isinstance(input_data, _SingleSizeImageList):
            raise ValueError("The given input ImageList contains images of different sizes.")  # noqa: TRY004
        if "column_name" not in kwargs or not isinstance(kwargs.get("column_name"), str):
            raise ValueError("The column_name is not set. The data can only be converted if the column_name is provided as `str` in the kwargs.")
        if "one_hot_encoder" not in kwargs or not isinstance(kwargs.get("one_hot_encoder"), OneHotEncoder):
            raise ValueError("The one_hot_encoder is not set. The data can only be converted if the one_hot_encoder is provided as `OneHotEncoder` in the kwargs.")
        one_hot_encoder: OneHotEncoder = kwargs["one_hot_encoder"]
        column_name: str = kwargs["column_name"]

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


class OutputConversionImageToTable(_OutputConversion[ImageList, ImageDataset[Table]], _OutputConversionImage):

    def _data_conversion(self, input_data: ImageList, output_data: Tensor, **kwargs: Any) -> ImageDataset[Table]:
        import torch

        if not isinstance(input_data, _SingleSizeImageList):
            raise ValueError("The given input ImageList contains images of different sizes.")  # noqa: TRY004
        if "column_names" not in kwargs or not isinstance(kwargs.get("column_names"), list) and all(isinstance(element, str) for element in kwargs["column_names"]):
            raise ValueError("The column_names are not set. The data can only be converted if the column_names are provided as `list[str]` in the kwargs.")
        column_names: list[str] = kwargs["column_names"]

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


class OutputConversionImageToImage(_OutputConversion[ImageList, ImageDataset[ImageList]], _OutputConversionImage):

    def _data_conversion(
        self, input_data: ImageList, output_data: Tensor, **kwargs: Any  # noqa: ARG002
    ) -> ImageDataset[ImageList]:
        import torch

        if not isinstance(input_data, _SingleSizeImageList):
            raise ValueError("The given input ImageList contains images of different sizes.")  # noqa: TRY004

        return ImageDataset[ImageList](input_data, _SingleSizeImageList._create_from_tensor((output_data * 255).to(torch.uint8), list(
            range(output_data.size(dim=0)))))
