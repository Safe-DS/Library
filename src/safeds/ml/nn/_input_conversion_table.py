from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.ml.nn._input_conversion import _InputConversion


class InputConversionTable(_InputConversion[TabularDataset, Table]):
    """The input conversion for a neural network, defines the input parameters for the neural network."""

    def __init__(self) -> None:
        """Define the input parameters for the neural network in the input conversion."""
        self._feature_names: list[str] = []
        self._target_name = ""
        self._time_name = ""

    @property
    def _data_size(self) -> int:
        return len(self._feature_names)

    def _data_conversion_fit(self, input_data: TabularDataset, batch_size: int, num_of_classes: int = 1) -> DataLoader:
        return input_data._into_dataloader_with_classes(
            batch_size,
            num_of_classes,
        )

    def _set_parameters(
        self,
        target_name: str,
        time_name: str,
        feature_names: list[str] | None = None,
    ) -> None:
        # time instance parameter won't be used, but is there for Linter
        self._time_name = time_name
        self._target_name = target_name

        self._feature_names = feature_names

    def _data_conversion_predict(self, input_data: Table, batch_size: int) -> DataLoader:
        return input_data._into_dataloader(batch_size)

    def _is_fit_data_valid(self, input_data: TabularDataset) -> bool:
        return (sorted(input_data.features.column_names)).__eq__(sorted(self._feature_names))

    def _is_predict_data_valid(self, input_data: Table) -> bool:
        return (sorted(input_data.column_names)).__eq__(sorted(self._feature_names))
