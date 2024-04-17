from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from torch import Tensor

from safeds.data.tabular.containers import TaggedTable, TimeSeries, Table

I = TypeVar('I', Table, TimeSeries)
O = TypeVar('O', TaggedTable, TimeSeries)


class _OutputConversion(Generic[I, O], ABC):
    """
    The output conversion for a neural network, defines the output parameters for the neural network.
    """

    @abstractmethod
    def _data_conversion(self, input_data: I, output_data: Tensor) -> O:
        pass
