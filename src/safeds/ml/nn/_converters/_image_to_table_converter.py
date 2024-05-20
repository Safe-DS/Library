from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from safeds._config import _init_default_device
from safeds._utils import _structural_hash
from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
from safeds.data.labeled.containers import ImageDataset
from safeds.data.labeled.containers._image_dataset import _TableAsTensor
from safeds.data.tabular.containers import Table
from safeds.ml.nn.typing import ConstantImageSize, VariableImageSize

from ._image_converter import _ImageConverter

if TYPE_CHECKING:
    from torch import Tensor

    from safeds.data.image.containers import ImageList


class _ImageToTableConverter(_ImageConverter):
    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, input_data: ImageDataset[Table], is_for_classification: bool) -> None:
        if is_for_classification:
            super().__init__(ConstantImageSize.from_image_size(input_data.input_size))
        else:
            super().__init__(VariableImageSize.from_image_size(input_data.input_size))

        self._output_type = type(input_data._output)
        self._output_size = len(input_data._output._column_names)
        self._column_names = input_data._output._column_names

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self is other) or (
            self._input_size == other._input_size
            and self._output_size == other._output_size
            and self._column_names == other._column_names
            and self._output_type == other._output_type
        )

    def __hash__(self) -> int:
        return _structural_hash(
            self.__class__.__name__,
            self._input_size,
            self._output_size,
            self._column_names,
            self._output_type,
        )

    def __sizeof__(self) -> int:
        return (
            sys.getsizeof(self._input_size)
            + sys.getsizeof(self._output_size)
            + sys.getsizeof(self._column_names)
            + sys.getsizeof(self._output_type)
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _data_conversion_output(self, input_data: ImageList, output_data: Tensor) -> ImageDataset[Table]:
        import torch

        _init_default_device()

        if not isinstance(input_data, _SingleSizeImageList):
            raise ValueError("The given input ImageList contains images of different sizes.")  # noqa: TRY004

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

    def _is_fit_data_valid(self, input_data: ImageDataset) -> bool:
        if not isinstance(input_data._output, self._output_type):
            return False
        if self._column_names != input_data._output._column_names:
            return False
        return input_data.input_size == self._input_size and input_data.output_size == self._output_size
