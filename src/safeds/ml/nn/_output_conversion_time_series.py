from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
from safeds.data.labeled.containers import TimeSeriesDataset
from safeds.data.tabular.containers import Column, Table
from safeds.ml.nn._output_conversion import _OutputConversion


class OutputConversionTimeSeries(_OutputConversion[TimeSeriesDataset, TimeSeriesDataset]):
    """The output conversion for a neural network, defines the output parameters for the neural network."""

    def __init__(self, prediction_name: str = "prediction_nn", window_size: int = 1, forecast_horizon: int = 1) -> None:
        """
        Define the output parameters for the neural network in the output conversion.

        Parameters
        ----------
        prediction_name
            The name of the new column where the prediction will be stored.
        """
        self._prediction_name = prediction_name
        self._window_size = window_size
        self._forecast_horizon = forecast_horizon

    def _data_conversion(self, input_data: TimeSeriesDataset, output_data: Tensor) -> TimeSeriesDataset:
        input_data_table = input_data.to_table()
        input_data_table = Table.from_rows(input_data_table.to_rows()[self._window_size + self._forecast_horizon:])

        return input_data_table.add_column(Column(self._prediction_name, output_data.tolist())).to_time_series_dataset(
            target_name=self._prediction_name,
            time_name=input_data.time.name,
            extra_names=input_data.extras.column_names,
        )
