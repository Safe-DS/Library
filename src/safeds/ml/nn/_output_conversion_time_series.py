from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from safeds.data.tabular.containers import Column, Table, TimeSeries
from safeds.ml.nn._output_conversion import _OutputConversion


class OutputConversionTimeSeries(_OutputConversion[TimeSeries, TimeSeries]):
    """The output conversion for a neural network, defines the output parameters for the neural network."""

    def __init__(self, prediction_name: str = "prediction_nn") -> None:
        """
        Define the output parameters for the neural network in the output conversion.

        Parameters
        ----------
        prediction_name
            The name of the new column where the prediction will be stored.
        """
        self._prediction_name = prediction_name

    def _data_conversion(self, input_data: TimeSeries, output_data: Tensor) -> TimeSeries:
        return input_data.add_column(Column(self._prediction_name, output_data.tolist())).time_columns(
            self._prediction_name,
            input_data.time.name,
            input_data.features.column_names
        )
