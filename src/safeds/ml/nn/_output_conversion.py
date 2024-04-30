from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar, Unpack, Any

from safeds.data.image.containers import ImageList
from safeds.data.labeled.containers import ImageDataset

if TYPE_CHECKING:
    from torch import Tensor

from safeds.data.tabular.containers import Table, TaggedTable, TimeSeries

IT = TypeVar("IT", Table, TimeSeries, ImageList)
OT = TypeVar("OT", TaggedTable, TimeSeries, ImageDataset)


class _OutputConversion(Generic[IT, OT], ABC):
    """The output conversion for a neural network, defines the output parameters for the neural network."""

    @abstractmethod
    def _data_conversion(self, input_data: IT, output_data: Tensor, **kwargs: Unpack[dict[str, Any]]) -> OT:
        pass  # pragma: no cover
