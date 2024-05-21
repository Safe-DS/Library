from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from safeds._config import _init_default_device
from safeds._utils import _structural_hash
from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
from safeds.data.labeled.containers import ImageDataset
from safeds.data.labeled.containers._image_dataset import _TableAsTensor
from safeds.data.tabular.containers import Table

from ._image_converter import _ImageConverter

if TYPE_CHECKING:
    from torch import Tensor

    from safeds.data.image.containers import ImageList
    from safeds.ml.nn.typing import ModelImageSize


class _ImageToTableConverter(_ImageConverter):
    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        image_size: ModelImageSize,
        column_names: list[str],
        *,
        output_size: int | None = None,
    ) -> None:
        super().__init__(image_size, output_size)

        self._column_names: list[str] = column_names

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _ImageToTableConverter):
            return NotImplemented
        if self is other:
            return True
        return (
            self._input_size == other._input_size
            and self._output_size == other._output_size
            and self._column_names == other._column_names
        )

    def __hash__(self) -> int:
        return _structural_hash(
            super().__hash__(),
            self._column_names,
        )

    def __sizeof__(self) -> int:
        return (
            super().__sizeof__()
            + sys.getsizeof(self._column_names)
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
        if not isinstance(input_data._output, _TableAsTensor):
            return False

        return (
            self._column_names == input_data._output._column_names
            and input_data.input_size == self._input_size
            and input_data.output_size == self._output_size
        )
