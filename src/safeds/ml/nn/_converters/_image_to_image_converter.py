from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._config import _init_default_device
from safeds._utils import _structural_hash
from safeds.data.image.containers import ImageList
from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
from safeds.data.labeled.containers import ImageDataset

from ._image_converter import _ImageConverter

if TYPE_CHECKING:
    from torch import Tensor

    from safeds.ml.nn.typing import ModelImageSize


class _ImageToImageConverter(_ImageConverter):
    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        input_size: ModelImageSize,
        *,
        output_size: int | ModelImageSize | None = None,
    ) -> None:
        super().__init__(input_size, output_size)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _ImageToImageConverter):
            return NotImplemented
        if self is other:
            return True
        return self._input_size == other._input_size and self._output_size == other._output_size

    def __hash__(self) -> int:
        return _structural_hash(
            super().__hash__(),
        )

    def __sizeof__(self) -> int:
        return super().__sizeof__()

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
        if not isinstance(input_data._output, ImageList):
            return False

        return input_data.input_size == self._input_size and input_data.output_size == self._output_size
