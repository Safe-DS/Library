from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from safeds._config import _init_default_device
from safeds._utils import _structural_hash
from safeds.data.image.containers import ImageList
from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
from safeds.data.labeled.containers import ImageDataset
from safeds.data.labeled.containers._image_dataset import _ColumnAsTensor, _TableAsTensor

from ._image_converter import _ImageConverter

if TYPE_CHECKING:
    from torch import Tensor

    from safeds.data.tabular.transformation import OneHotEncoder
    from safeds.ml.nn.typing import ModelImageSize


class _ImageToImageConverter(_ImageConverter):
    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, image_size: ModelImageSize) -> None:
        self._input_size = image_size
        self._output_size: ModelImageSize | int | None = None
        self._one_hot_encoder: OneHotEncoder | None = None
        self._column_name: str | None = None
        self._column_names: list[str] | None = None
        self._output_type: type | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self is other) or (
            self._input_size == other._input_size
            and self._output_size == other._output_size
            and self._one_hot_encoder == other._one_hot_encoder
            and self._column_name == other._column_name
            and self._column_names == other._column_names
            and self._output_type == other._output_type
        )

    def __hash__(self) -> int:
        return _structural_hash(
            self.__class__.__name__,
            self._input_size,
            self._output_size,
            self._one_hot_encoder,
            self._column_name,
            self._column_names,
            self._output_type,
        )

    def __sizeof__(self) -> int:
        return (
            sys.getsizeof(self._input_size)
            + sys.getsizeof(self._output_size)
            + sys.getsizeof(self._one_hot_encoder)
            + sys.getsizeof(self._column_name)
            + sys.getsizeof(self._column_names)
            + sys.getsizeof(self._output_type)
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _data_conversion_output(
        self,
        input_data: ImageList,
        output_data: Tensor,
    ) -> ImageDataset[ImageList]:
        import torch

        _init_default_device()

        if not isinstance(input_data, _SingleSizeImageList):
            raise ValueError("The given input ImageList contains images of different sizes.")  # noqa: TRY004

        return ImageDataset[ImageList](
            input_data,
            _SingleSizeImageList._create_from_tensor(
                (output_data * 255).to(torch.uint8),
                list(range(output_data.size(dim=0))),
            ),
        )

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
