from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.data.tabular.containers import Column, Table

from ._converter import _Converter

if TYPE_CHECKING:
    from torch import Tensor
    from torch.utils.data import DataLoader

    from safeds.data.labeled.containers import TabularDataset


class _TableConverter(_Converter[Table, Column]):
    """
    The input conversion for a neural network, defines the input parameters for the neural network.

    prediction_name:
        The name of the new column where the prediction will be stored.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        input_data: TabularDataset,
        *,
        prediction_name: str = "prediction",
    ) -> None:
        self._target_name = input_data.target.name
        self._feature_names: list[str] = input_data.features.column_names
        self._prediction_name = prediction_name  # TODO: use target name, override existing column

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _TableConverter):
            return False
        if self is other:
            return True
        return (
            self._target_name == other._target_name
            and self._feature_names == other._feature_names
            and self._prediction_name == other._prediction_name
        )

    def __hash__(self) -> int:
        return _structural_hash(
            self.__class__.__name__,
            self._target_name,
            self._feature_names,
            self._prediction_name,
        )

    def __sizeof__(self) -> int:
        return sum(
            map(sys.getsizeof, (self._target_name, self._feature_names, self._prediction_name)),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def _data_size(self) -> int:
        return len(self._feature_names)

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _data_conversion_fit(self, input_data: TabularDataset, batch_size: int, num_of_classes: int = 1) -> DataLoader:
        return input_data._into_dataloader_with_classes(
            batch_size,
            num_of_classes,
        )

    def _data_conversion_predict(self, input_data: Table, batch_size: int) -> DataLoader:
        return input_data._into_dataloader(batch_size)

    def _data_conversion_output(self, input_data: Table, output_data: Tensor) -> TabularDataset:
        return input_data.add_columns([Column(self._prediction_name, output_data.tolist())]).to_tabular_dataset(
            self._prediction_name,
        )

    def _is_fit_data_valid(self, input_data: TabularDataset) -> bool:
        return sorted(input_data.features.column_names) == sorted(self._feature_names)

    def _is_predict_data_valid(self, input_data: Table) -> bool:
        return sorted(input_data.column_names) == sorted(self._feature_names)
