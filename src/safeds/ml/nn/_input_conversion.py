from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
from safeds.data.image.typing import ImageSize

from safeds.data.image.containers import ImageList
from safeds.data.labeled.containers import ImageDataset, TabularDataset, TimeSeriesDataset
from safeds.data.tabular.containers import Table

FT = TypeVar("FT", TabularDataset, TimeSeriesDataset, ImageDataset)
PT = TypeVar("PT", Table, TimeSeriesDataset, ImageList)


class InputConversion(Generic[FT, PT], ABC):
    """The input conversion for a neural network, defines the input parameters for the neural network."""

    @property
    @abstractmethod
    def _data_size(self) -> int | ImageSize:
        pass  # pragma: no cover

    @abstractmethod
    def _data_conversion_fit(
        self,
        input_data: FT,
        batch_size: int,
        num_of_classes: int = 1,
    ) -> DataLoader | ImageDataset:
        pass  # pragma: no cover

    @abstractmethod
    def _data_conversion_predict(self, input_data: PT, batch_size: int) -> DataLoader | _SingleSizeImageList:
        pass  # pragma: no cover

    @abstractmethod
    def _is_fit_data_valid(self, input_data: FT) -> bool:
        pass  # pragma: no cover

    @abstractmethod
    def _is_predict_data_valid(self, input_data: PT) -> bool:
        pass  # pragma: no cover

    @abstractmethod
    def _get_output_configuration(self) -> dict[str, Any]:
        pass  # pragma: no cover
