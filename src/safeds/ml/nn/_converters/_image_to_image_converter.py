from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from safeds._config import _init_default_device
from safeds._utils import _structural_hash
from safeds.data.image.containers import ImageList
from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
from safeds.data.labeled.containers import ImageDataset
from safeds.ml.nn.typing import ConstantImageSize, VariableImageSize

from ._image_converter import _ImageConverter

if TYPE_CHECKING:
    from torch import Tensor


class _ImageToImageConverter(_ImageConverter):
    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, input_data: ImageDataset[ImageList], is_for_classification: bool) -> None:
        if is_for_classification:
            super().__init__(ConstantImageSize.from_image_size(input_data.input_size))
        else:
            super().__init__(VariableImageSize.from_image_size(input_data.input_size))

        self._output_type = type(input_data._output)
        self._output_size = input_data.output_size

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self is other) or (
            self._input_size == other._input_size
            and self._output_size == other._output_size
            and self._output_type == other._output_type
        )

    def __hash__(self) -> int:
        return _structural_hash(
            self.__class__.__name__,
            self._input_size,
            self._output_size,
            self._output_type,
        )

    def __sizeof__(self) -> int:
        return sys.getsizeof(self._input_size) + sys.getsizeof(self._output_size) + sys.getsizeof(self._output_type)

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
        if not isinstance(input_data._output, self._output_type):
            return False
        return input_data.input_size == self._input_size and input_data.output_size == self._output_size
