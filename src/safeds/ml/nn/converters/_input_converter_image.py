from __future__ import annotations

import sys
from abc import ABC
from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.data.image.containers import ImageList
from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
from safeds.data.labeled.containers import ImageDataset
from safeds.data.labeled.containers._image_dataset import _ColumnAsTensor, _TableAsTensor

from ._input_converter import InputConversion

if TYPE_CHECKING:
    from safeds.data.tabular.transformation import OneHotEncoder
    from safeds.ml.nn.typing import ModelImageSize


class _InputConversionImage(InputConversion[ImageDataset, ImageList], ABC):
    """
    The input conversion for a neural network, defines the input parameters for the neural network.

    Parameters
    ----------
    image_size:
        the size of the input images
    """

    def __init__(self, image_size: ModelImageSize) -> None:
        self._input_size = image_size
        self._output_size: ModelImageSize | int | None = None
        self._one_hot_encoder: OneHotEncoder | None = None
        self._column_name: str | None = None
        self._column_names: list[str] | None = None
        self._output_type: type | None = None

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

    def __sizeof__(self) -> int:
        return (
            sys.getsizeof(self._input_size)
            + sys.getsizeof(self._output_size)
            + sys.getsizeof(self._one_hot_encoder)
            + sys.getsizeof(self._column_name)
            + sys.getsizeof(self._column_names)
            + sys.getsizeof(self._output_type)
        )

    @property
    def _data_size(self) -> ModelImageSize:
        return self._input_size

    def _data_conversion_fit(
        self,
        input_data: ImageDataset,
        batch_size: int,  # noqa: ARG002
        num_of_classes: int = 1,  # noqa: ARG002
    ) -> ImageDataset:
        return input_data

    def _data_conversion_predict(self, input_data: ImageList, batch_size: int) -> _SingleSizeImageList:  # noqa: ARG002
        return input_data._as_single_size_image_list()

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

    def _is_predict_data_valid(self, input_data: ImageList) -> bool:
        return isinstance(input_data, _SingleSizeImageList) and input_data.sizes[0] == self._input_size
