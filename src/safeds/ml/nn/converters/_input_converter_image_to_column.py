from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._config import _init_default_device
from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
from safeds.data.labeled.containers import ImageDataset
from safeds.data.labeled.containers._image_dataset import _ColumnAsTensor
from safeds.data.tabular.containers import Column

from ._input_converter_image import _InputConversionImage

if TYPE_CHECKING:
    from torch import Tensor

    from safeds.data.image.containers import ImageList
    from safeds.data.tabular.transformation import OneHotEncoder


class InputConversionImageToColumn(_InputConversionImage):
    def _data_conversion_output(
        self,
        input_data: ImageList,
        output_data: Tensor,
    ) -> ImageDataset[Column]:
        import torch

        _init_default_device()

        if not isinstance(input_data, _SingleSizeImageList):
            raise ValueError("The given input ImageList contains images of different sizes.")  # noqa: TRY004
        if self._column_name is None:
            raise ValueError(
                "The column_name is not set. The data can only be converted if the column_name is provided as `str` in the kwargs.",
            )
        if self._one_hot_encoder is None:
            raise ValueError(
                "The one_hot_encoder is not set. The data can only be converted if the one_hot_encoder is provided as `OneHotEncoder` in the kwargs.",
            )
        one_hot_encoder: OneHotEncoder = self._one_hot_encoder
        column_name: str = self._column_name

        output = torch.zeros(len(input_data), len(one_hot_encoder._get_names_of_added_columns()))
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
