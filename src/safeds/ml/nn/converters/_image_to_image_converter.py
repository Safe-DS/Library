from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._config import _init_default_device
from safeds.data.image.containers import ImageList
from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
from safeds.data.labeled.containers import ImageDataset

from ._image_converter import _ImageConverter

if TYPE_CHECKING:
    from torch import Tensor


class _ImageToImageConverter(_ImageConverter):
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
