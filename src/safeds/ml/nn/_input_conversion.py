from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from safeds.data.image.typing import ImageSize

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

from safeds.data.tabular.containers import Table, TaggedTable, TimeSeries
from safeds.data.image.containers import ImageList
from safeds.data.labeled.containers import ImageDataset

FT = TypeVar("FT", TaggedTable, TimeSeries)
PT = TypeVar("PT", Table, TimeSeries)


class _InputConversion(Generic[FT, PT], ABC):
    """The input conversion for a neural network, defines the input parameters for the neural network."""

    @property
    @abstractmethod
    def _data_size(self) -> int | ImageSize:
        pass  # pragma: no cover

    @abstractmethod
    def _data_conversion_fit(self, input_data: FT, batch_size: int, num_of_classes: int = 1) -> DataLoader | ImageDataset:
        pass  # pragma: no cover

    @abstractmethod
    def _data_conversion_predict(self, input_data: PT, batch_size: int) -> DataLoader | ImageList:
        pass  # pragma: no cover

    @abstractmethod
    def _is_fit_data_valid(self, input_data: FT) -> bool:
        pass  # pragma: no cover

    @abstractmethod
    def _is_predict_data_valid(self, input_data: PT) -> bool:
        pass  # pragma: no cover
