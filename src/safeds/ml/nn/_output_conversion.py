from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from torch import Tensor

from safeds.data.tabular.containers import TaggedTable, TimeSeries, Table

IT = TypeVar("IT", Table, TimeSeries)
OT = TypeVar("OT", TaggedTable, TimeSeries)


class _OutputConversion(Generic[IT, OT], ABC):
    """The output conversion for a neural network, defines the output parameters for the neural network."""

    @abstractmethod
    def _data_conversion(self, input_data: IT, output_data: Tensor) -> OT:
        pass
