from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.data.image.containers import ImageList
from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
from safeds.data.labeled.containers import ImageDataset

from ._converter import _Converter

if TYPE_CHECKING:
    from safeds.ml.nn.typing import ModelImageSize


class _ImageConverter(_Converter[ImageDataset, ImageList], ABC):
    """
    The input conversion for a neural network, defines the input parameters for the neural network.

    Parameters
    ----------
    image_size:
        the size of the input images
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def __init__(
        self,
        image_size: ModelImageSize,
        output_size: int | ModelImageSize | None,
        output_type: type | None,
    ) -> None:
        self._input_size: ModelImageSize = image_size
        self._output_size: int | ModelImageSize | None = None
        self._output_type: type | None = None

    @abstractmethod
    def __hash__(self) -> int:
        return _structural_hash(
            self.__class__.__name__,
            self._input_size,
            self._output_size,
            self._output_type,
        )

    @abstractmethod
    def __sizeof__(self) -> int:
        return (
            sys.getsizeof(self._input_size)
            + sys.getsizeof(self._output_size)
            + sys.getsizeof(self._output_type)
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def _data_size(self) -> ModelImageSize:
        return self._input_size

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _data_conversion_fit(
        self,
        input_data: ImageDataset,
        batch_size: int,  # noqa: ARG002
        num_of_classes: int = 1,  # noqa: ARG002
    ) -> ImageDataset:
        return input_data

    def _data_conversion_predict(self, input_data: ImageList, batch_size: int) -> _SingleSizeImageList:  # noqa: ARG002
        return input_data._as_single_size_image_list()

    def _is_predict_data_valid(self, input_data: ImageList) -> bool:
        return isinstance(input_data, _SingleSizeImageList) and input_data.sizes[0] == self._input_size
