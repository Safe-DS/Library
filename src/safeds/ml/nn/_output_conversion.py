from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from torch import Tensor

from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table, TimeSeries

IT = TypeVar("IT", Table, TimeSeries)
OT = TypeVar("OT", TabularDataset, TimeSeries)


class OutputConversion(Generic[IT, OT], ABC):
    """The output conversion for a neural network, defines the output parameters for the neural network."""

    @abstractmethod
    def _data_conversion(self, input_data: IT, output_data: Tensor) -> OT:
        pass  # pragma: no cover
