from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.data.labeled.containers import TimeSeriesDataset
from safeds.data.tabular.containers import Column, Table

from ._input_converter import InputConversion

if TYPE_CHECKING:
    from torch import Tensor
    from torch.utils.data import DataLoader


class InputConversionTimeSeries(InputConversion[TimeSeriesDataset, Table]):
    """The input conversion for a neural network, defines the input parameters for the neural network."""

    def __init__(
        self,
    ) -> None:
        self._window_size = 0
        self._forecast_horizon = 0
        self._first = True
        self._target_name: str = ""
        self._feature_names: list[str] = []
        self._continuous: bool = False
        self._extra_names: list[str] = []

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self is other) or (
            self._window_size == other._window_size
            and self._forecast_horizon == other._forecast_horizon
            and self._first == other._first
            and self._target_name == other._target_name
            and self._feature_names == other._feature_names
            and self._continuous == other._continuous
            and self._extra_names == other._extra_names
        )

    def __hash__(self) -> int:
        return _structural_hash(
            self.__class__.__name__,
            self._window_size,
            self._forecast_horizon,
            self._target_name,
            self._feature_names,
            self._continuous,
            self._extra_names,
        )

    def __sizeof__(self) -> int:
        return (
            sys.getsizeof(self._window_size)
            + sys.getsizeof(self._forecast_horizon)
            + sys.getsizeof(self._target_name)
            + sys.getsizeof(self._feature_names)
            + sys.getsizeof(self._continuous)
            + sys.getsizeof(self._extra_names)
        )

    @property
    def _data_size(self) -> int:
        """
        Gives the size for the input of an internal layer.

        Returns
        -------
        size:
            The size of the input for the neural network
        """
        return (len(self._feature_names) + 1) * self._window_size

    def _data_conversion_fit(
        self,
        input_data: TimeSeriesDataset,
        batch_size: int,
        num_of_classes: int = 1,
    ) -> DataLoader:
        self._num_of_classes = num_of_classes
        return input_data._into_dataloader_with_window(
            self._window_size,
            self._forecast_horizon,
            batch_size,
            continuous=self._continuous,
        )

    def _data_conversion_predict(self, input_data: Table, batch_size: int) -> DataLoader:
        data: TimeSeriesDataset
        data = input_data.to_time_series_dataset(
            target_name=self._target_name,
            window_size=self._window_size,
            extra_names=self._extra_names,
            forecast_horizon=self._forecast_horizon,
            continuous=self._continuous,
        )
        return data._into_dataloader_with_window_predict(
            self._window_size,
            self._forecast_horizon,
            batch_size,
        )

    def _data_conversion_output(
        self,
        input_data: Table,
        output_data: Tensor,
    ) -> TimeSeriesDataset:
        table_data: Table
        window_size: int = self._window_size
        forecast_horizon: int = self._forecast_horizon
        table_data = input_data
        input_data_table = table_data.slice_rows(start=window_size + forecast_horizon)

        return input_data_table.replace_column(
            self._target_name,
            [Column(self._target_name, output_data.tolist())],
        ).to_time_series_dataset(
            target_name=self._target_name,
            window_size=self._window_size,
            extra_names=self._extra_names,
            forecast_horizon=self._forecast_horizon,
            continuous=self._continuous,
        )

    def _is_fit_data_valid(self, input_data: TimeSeriesDataset) -> bool:
        if self._first:
            self._window_size = input_data.window_size
            self._forecast_horizon = input_data.forecast_horizon
            self._feature_names = input_data.features.column_names
            self._target_name = input_data.target.name
            self._continuous = input_data._continuous
            self._first = False
            self._extra_names = input_data.extras.column_names
        return (
            sorted(input_data.features.column_names).__eq__(
                sorted(self._feature_names),
            )
            and input_data.target.name == self._target_name
        )

    def _is_predict_data_valid(self, input_data: Table) -> bool:
        for name in self._feature_names:
            if name not in input_data.column_names:
                return False
        if self._target_name not in input_data.column_names:
            return False
        return True
