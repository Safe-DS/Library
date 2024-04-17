from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

from safeds.data.tabular.containers import Table, TimeSeries

T = TypeVar("T", Table, TimeSeries)


class _InputConversion(Generic[T], ABC):
    """The input conversion for a neural network, defines the input parameters for the neural network."""

    @property
    @abstractmethod
    def _data_size(self) -> int:
        pass  # pragma: no cover

    @abstractmethod
    def _data_conversion_fit(self, input_data: T, batch_size: int, num_of_classes: int = 1) -> DataLoader:
        pass  # pragma: no cover

    @abstractmethod
    def _data_conversion_predict(self, input_data: T, batch_size: int) -> DataLoader:
        pass  # pragma: no cover

    @abstractmethod
    def _is_data_valid(self, input_data: T) -> bool:
        pass  # pragma: no cover
