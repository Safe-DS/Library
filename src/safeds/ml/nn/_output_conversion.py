from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from safeds.data.image.containers import ImageList
from safeds.data.labeled.containers import ImageDataset, TabularDataset

if TYPE_CHECKING:
    from torch import Tensor

from safeds.data.labeled.containers import TimeSeriesDataset
from safeds.data.tabular.containers import Table

IT = TypeVar("IT", Table, TimeSeriesDataset, ImageList)
OT = TypeVar("OT", TabularDataset, TimeSeriesDataset, ImageDataset)


class OutputConversion(Generic[IT, OT], ABC):
    """The output conversion for a neural network, defines the output parameters for the neural network."""

    @abstractmethod
    def _data_conversion(self, input_data: IT, output_data: Tensor, **kwargs: Any) -> OT:
        pass  # pragma: no cover
