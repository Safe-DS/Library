from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._config import _init_default_device
from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
from safeds.data.labeled.containers import ImageDataset
from safeds.data.labeled.containers._image_dataset import _TableAsTensor
from safeds.data.tabular.containers import Table

from ._input_converter_image import _InputConversionImage

if TYPE_CHECKING:
    from torch import Tensor

    from safeds.data.image.containers import ImageList


class InputConversionImageToTable(_InputConversionImage):
    def _data_conversion_output(self, input_data: ImageList, output_data: Tensor) -> ImageDataset[Table]:
        import torch

        _init_default_device()

        if not isinstance(input_data, _SingleSizeImageList):
            raise ValueError("The given input ImageList contains images of different sizes.")  # noqa: TRY004
        if self._column_names is None:
            raise ValueError(
                "The column_names are not set. The data can only be converted if the column_names are provided as `list[str]` in the kwargs.",
            )
        column_names: list[str] = self._column_names

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
