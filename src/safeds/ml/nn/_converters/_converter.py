from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from safeds.data.image.containers import ImageList
from safeds.data.tabular.containers import Column, Table

if TYPE_CHECKING:
    from torch import Tensor
    from torch.utils.data import DataLoader

    from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
    from safeds.data.labeled.containers import Dataset, ImageDataset
    from safeds.ml.nn.typing import ModelImageSize

In = TypeVar("In", ImageList, Table)
Out = TypeVar("Out", Column, ImageList, Table)


class _Converter(Generic[In, Out], ABC):
    """The input conversion for a neural network, defines the input parameters for the neural network."""

    # TODO: add docstring for properties and template methods

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def __eq__(self, other: object) -> bool: ...

    @abstractmethod
    def __hash__(self) -> int: ...

    @abstractmethod
    def __sizeof__(self) -> int: ...

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    @abstractmethod
    def _data_size(self) -> int | ModelImageSize: ...

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def _data_conversion_fit(
        self,
        input_data: Dataset[In, Out],
        batch_size: int,
        num_of_classes: int = 1,
    ) -> DataLoader | ImageDataset: ...  # TODO: unify return type (data loader)

    @abstractmethod
    # TODO: unify return type (data loader)
    def _data_conversion_predict(
        self,
        input_data: In | Dataset[In, Out],
        batch_size: int,
    ) -> DataLoader | _SingleSizeImageList: ...

    @abstractmethod
    def _data_conversion_output(self, input_data: In | Dataset[In, Out], output_data: Tensor) -> Dataset[In, Out]: ...

    @abstractmethod
    def _is_fit_data_valid(self, input_data: Dataset[In, Out]) -> bool: ...

    @abstractmethod
    def _is_predict_data_valid(self, input_data: In | Dataset[In, Out]) -> bool: ...
