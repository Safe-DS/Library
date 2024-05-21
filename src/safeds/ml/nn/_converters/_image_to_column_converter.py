from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from safeds._config import _init_default_device
from safeds._utils import _structural_hash
from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
from safeds.data.labeled.containers import ImageDataset
from safeds.data.labeled.containers._image_dataset import _ColumnAsTensor, _TableAsTensor
from safeds.data.tabular.containers import Column

from ._image_converter import _ImageConverter

if TYPE_CHECKING:
    from torch import Tensor

    from safeds.data.image.containers import ImageList
    from safeds.data.tabular.transformation import OneHotEncoder
    from safeds.ml.nn.typing import ModelImageSize


class _ImageToColumnConverter(_ImageConverter):
    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, image_size: ModelImageSize) -> None:
        super().__init__(image_size)

        self._output_size: ModelImageSize | int | None = None
        self._output_type: type | None = None
        self._one_hot_encoder: OneHotEncoder | None = None
        self._column_name: str | None = None
        self._column_names: list[str] | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        if self is other:
            return True
        return (
            self._input_size == other._input_size
            and self._output_size == other._output_size
            and self._output_type == other._output_type
            and self._one_hot_encoder == other._one_hot_encoder
            and self._column_name == other._column_name
            and self._column_names == other._column_names
        )

    def __hash__(self) -> int:
        return _structural_hash(
            super().__hash__(),
            self._output_size,
            self._output_type,
            self._one_hot_encoder,
            self._column_name,
            self._column_names,
        )

    def __sizeof__(self) -> int:
        return (
            super().__sizeof__()
            + sys.getsizeof(self._output_size)
            + sys.getsizeof(self._output_type)
            + sys.getsizeof(self._one_hot_encoder)
            + sys.getsizeof(self._column_name)
            + sys.getsizeof(self._column_names)
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

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
                "The column_name is not set. "
                "The data can only be converted if the column_name is provided as `str` in the kwargs.",
            )
        if self._one_hot_encoder is None:
            raise ValueError(
                "The one_hot_encoder is not set. "
                "The data can only be converted if the one_hot_encoder is provided as `OneHotEncoder` in the kwargs.",
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

    def _is_fit_data_valid(self, input_data: ImageDataset) -> bool:
        if self._output_type is None:
            self._output_type = type(input_data._output)
            self._output_size = input_data.output_size
        elif not isinstance(input_data._output, self._output_type):
            return False
        if isinstance(input_data._output, _ColumnAsTensor):
            if self._column_name is None and self._one_hot_encoder is None:
                self._one_hot_encoder = input_data._output._one_hot_encoder
                self._column_name = input_data._output._column_name
            elif (
                self._column_name != input_data._output._column_name
                or self._one_hot_encoder != input_data._output._one_hot_encoder
            ):
                return False
        elif isinstance(input_data._output, _TableAsTensor):
            if self._column_names is None:
                self._column_names = input_data._output._column_names
            elif self._column_names != input_data._output._column_names:
                return False
        return input_data.input_size == self._input_size and input_data.output_size == self._output_size
