from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, TypeVar

from safeds.data.image.containers import ImageList
from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
from safeds.data.tabular.containers import Column, Table

from ._converter import _Converter

if TYPE_CHECKING:
    from safeds.data.labeled.containers import ImageDataset
    from safeds.ml.nn.typing import ModelImageSize

Out = TypeVar("Out", Column, ImageList, Table)


class _ImageConverter(_Converter[ImageList, Out], ABC):
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

    def __init__(self, image_size: ModelImageSize) -> None:
        self._input_size = image_size

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
