from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from safeds.data.tabular.containers import Column, Table, TimeSeries
from safeds.ml.nn._output_conversion import _OutputConversion


class OutputConversionTimeSeries(_OutputConversion[TimeSeries, TimeSeries]):
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

    def _data_conversion(self, input_data: TimeSeries, output_data: Tensor) -> TimeSeries:
        input_data_table = Table.from_rows(input_data.to_rows()[self._window_size + self._forecast_horizon :])

        return input_data_table.add_column(Column(self._prediction_name, output_data.tolist())).time_columns(
            self._prediction_name,
            input_data.time.name,
            input_data.features.column_names,
        )
