from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

from safeds.data.labeled.containers import TabularDataset, TimeSeriesDataset
from safeds.data.tabular.containers import Table

FT = TypeVar("FT", TabularDataset, TimeSeriesDataset)
PT = TypeVar("PT", Table, TimeSeriesDataset)


class InputConversion(Generic[FT, PT], ABC):
    """The input conversion for a neural network, defines the input parameters for the neural network."""

    @property
    @abstractmethod
    def _data_size(self) -> int:
        pass  # pragma: no cover

    @abstractmethod
    def _data_conversion_fit(self, input_data: FT, batch_size: int, num_of_classes: int = 1) -> DataLoader:
        pass  # pragma: no cover

    @abstractmethod
    def _data_conversion_predict(self, input_data: PT, batch_size: int) -> DataLoader:
        pass  # pragma: no cover

    @abstractmethod
    def _set_parameters(
        self,
        target_name: str,
        time_name: str,
        feature_names: list[str],
    ) -> None:
        pass  # pragma: no cover

    @abstractmethod
    def _is_fit_data_valid(self, input_data: FT) -> bool:
        pass  # pragma: no cover

    @abstractmethod
    def _is_predict_data_valid(self, input_data: PT) -> bool:
        pass  # pragma: no cover
