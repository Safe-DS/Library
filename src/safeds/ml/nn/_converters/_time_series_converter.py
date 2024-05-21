from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from safeds._utils import _structural_hash
from safeds.data.labeled.containers import TimeSeriesDataset
from safeds.data.tabular.containers import Column

from ._converter import _Converter

if TYPE_CHECKING:
    from torch import Tensor
    from torch.utils.data import DataLoader


class _TimeSeriesConverter(_Converter[TimeSeriesDataset, TimeSeriesDataset]):
    """The input conversion for a neural network, defines the input parameters for the neural network."""

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        *,
        prediction_name: str = "prediction_nn",
    ) -> None:
        self._window_size = 0
        self._forecast_horizon = 0
        self._first = True
        self._target_name: str = ""
        self._time_name: str = ""
        self._feature_names: list[str] = []
        self._prediction_name = prediction_name  # TODO: use target name, override existing column

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _TimeSeriesConverter):
            return False
        return self._prediction_name == other._prediction_name

    def __hash__(self) -> int:
        return _structural_hash(self.__class__.__name__ + self._prediction_name)

    def __sizeof__(self) -> int:
        return sys.getsizeof(self._prediction_name)

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

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

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

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
        )

    def _data_conversion_predict(self, input_data: TimeSeriesDataset, batch_size: int) -> DataLoader:
        return input_data._into_dataloader_with_window_predict(self._window_size, self._forecast_horizon, batch_size)

    def _data_conversion_output(
        self,
        input_data: TimeSeriesDataset,
        output_data: Tensor,
    ) -> TimeSeriesDataset:
        window_size: int = self._window_size
        forecast_horizon: int = self._forecast_horizon
        input_data_table = input_data.to_table()
        input_data_table = input_data_table.slice_rows(start=window_size + forecast_horizon)

        return input_data_table.add_columns(
            [Column(self._prediction_name, output_data.tolist())],
        ).to_time_series_dataset(
            target_name=self._prediction_name,
            time_name=input_data.time.name,
            extra_names=input_data.extras.column_names,
            window_size=window_size,
        )

    def _is_fit_data_valid(self, input_data: TimeSeriesDataset) -> bool:
        if self._first:
            self._window_size = input_data.window_size
            self._forecast_horizon = input_data.forecast_horizon
            self._time_name = input_data.time.name
            self._feature_names = input_data.features.column_names
            self._target_name = input_data.target.name
            self._first = False
        return (
            (sorted(input_data.features.column_names)).__eq__(sorted(self._feature_names))
            and input_data.target.name == self._target_name
            and input_data.time.name == self._time_name
        )

    def _is_predict_data_valid(self, input_data: TimeSeriesDataset) -> bool:
        return self._is_fit_data_valid(input_data)
