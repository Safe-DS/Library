from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from safeds.data.image.containers import ImageList
from safeds.data.labeled.containers import ImageDataset, TabularDataset, TimeSeriesDataset
from safeds.data.tabular.containers import Table

if TYPE_CHECKING:
    from torch import Tensor
    from torch.utils.data import DataLoader

    from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
    from safeds.ml.nn.typing import ModelImageSize

FT = TypeVar("FT", TabularDataset, TimeSeriesDataset, ImageDataset)
PT = TypeVar("PT", Table, TimeSeriesDataset, ImageList)


class InputConversion(Generic[FT, PT], ABC):
    """The input conversion for a neural network, defines the input parameters for the neural network."""

    @property
    @abstractmethod
    def _data_size(self) -> int | ModelImageSize: ...

    @abstractmethod
    def _data_conversion_fit(
        self,
        input_data: FT,
        batch_size: int,
        num_of_classes: int = 1,
    ) -> DataLoader | ImageDataset: ...

    @abstractmethod
    def _data_conversion_predict(self, input_data: PT, batch_size: int) -> DataLoader | _SingleSizeImageList: ...

    @abstractmethod
    def _data_conversion_output(self, input_data: PT, output_data: Tensor) -> FT: ...

    @abstractmethod
    def _is_fit_data_valid(self, input_data: FT) -> bool: ...

    @abstractmethod
    def _is_predict_data_valid(self, input_data: PT) -> bool: ...
