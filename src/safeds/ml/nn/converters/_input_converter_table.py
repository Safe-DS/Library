from __future__ import annotations

from typing import TYPE_CHECKING

from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Column, Table
from safeds.exceptions import InvalidFitDataError

from ._input_converter import InputConversion

if TYPE_CHECKING:
    from torch import Tensor
    from torch.utils.data import DataLoader


class InputConversionTable(InputConversion[TabularDataset, Table]):
    """The input conversion for a neural network, defines the input parameters for the neural network."""

    def __init__(self) -> None:
        self._target_name = ""
        self._feature_names: list[str] = []
        self._first = True

    @property
    def _data_size(self) -> int:
        return len(self._feature_names)

    def _data_conversion_fit(self, input_data: TabularDataset, batch_size: int, num_of_classes: int = 1) -> DataLoader:
        return input_data._into_dataloader_with_classes(
            batch_size,
            num_of_classes,
        )

    def _data_conversion_predict(self, input_data: Table, batch_size: int) -> DataLoader:
        return input_data._into_dataloader(batch_size)

    def _data_conversion_output(self, input_data: Table, output_data: Tensor) -> TabularDataset:
        return input_data.add_columns([Column(self._target_name, output_data.tolist())]).to_tabular_dataset(
            self._target_name,
        )

    def _is_fit_data_valid(self, input_data: TabularDataset) -> bool:
        if self._first:
            self._feature_names = input_data.features.column_names
            self._target_name = input_data.target.name
            self._first = False

            columns_with_missing_values = []
            columns_with_non_numerical_data = []

            for col in input_data.features.add_columns([input_data.target]).to_columns():
                if col.missing_value_count() > 0:
                    columns_with_missing_values.append(col.name)
                if not col.type.is_numeric:
                    columns_with_non_numerical_data.append(col.name)

            reason = ""
            if len(columns_with_missing_values) > 0:
                reason += f"The following Columns contain missing values: {columns_with_missing_values}\n"
            if len(columns_with_non_numerical_data) > 0:
                reason += f"The following Columns contain non-numerical data: {columns_with_non_numerical_data}"
            if reason != "":
                raise InvalidFitDataError(reason)

        return (sorted(input_data.features.column_names)).__eq__(sorted(self._feature_names))

    def _is_predict_data_valid(self, input_data: Table) -> bool:
        return (sorted(input_data.column_names)).__eq__(sorted(self._feature_names))
