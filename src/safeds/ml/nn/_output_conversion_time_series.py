from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from safeds._utils import _structural_hash

if TYPE_CHECKING:
    from torch import Tensor
from safeds.data.labeled.containers import TimeSeriesDataset
from safeds.data.tabular.containers import Column, Table
from safeds.ml.nn._output_conversion import OutputConversion


class OutputConversionTimeSeries(OutputConversion[TimeSeriesDataset, TimeSeriesDataset]):
    """The output conversion for a neural network, defines the output parameters for the neural network."""

    def __hash__(self) -> int:
        """
        Return a deterministic hash value for this OutputConversionTimeSeries instance.

        Returns
        -------
        hash:
            the hash value
        """
        return _structural_hash(self.__class__.__name__ + self._prediction_name)

    def __eq__(self, other: object) -> bool:
        """
        Compare two OutputConversionTimeSeries instances.

        Parameters
        ----------
        other:
            The OutputConversionTimeSeries instance to compare to.

        Returns
        -------
        equals:
            Whether the instances are the same.
        """
        if not isinstance(other, OutputConversionTimeSeries):
            return False
        return self._prediction_name == other._prediction_name

    def __sizeof__(self) -> int:
        """
        Return the complete size of this object.

        Returns
        -------
        size:
            Size of this object in bytes.
        """
        return sys.getsizeof(self._prediction_name)

    def __init__(self, prediction_name: str = "prediction_nn") -> None:
        """
        Define the output parameters for the neural network in the output conversion.

        Parameters
        ----------
        prediction_name:
            The name of the new column where the prediction will be stored.
        """
        self._prediction_name = prediction_name

    def _data_conversion(self, input_data: TimeSeriesDataset, output_data: Tensor, **kwargs: Any) -> TimeSeriesDataset:
        if "window_size" not in kwargs or not isinstance(kwargs.get("window_size"), int):
            raise ValueError(
                "The window_size is not set. The data can only be converted if the window_size is provided as `int` in the kwargs.",
            )
        if "forecast_horizon" not in kwargs or not isinstance(kwargs.get("forecast_horizon"), int):
            raise ValueError(
                "The forecast_horizon is not set. The data can only be converted if the forecast_horizon is provided as `int` in the kwargs.",
            )
        window_size: int = kwargs["window_size"]
        forecast_horizon: int = kwargs["forecast_horizon"]
        input_data_table = input_data.to_table()
        input_data_table = Table.from_rows(input_data_table.to_rows()[window_size + forecast_horizon :])

        return input_data_table.add_columns(
            [Column(self._prediction_name, output_data.tolist())]
        ).to_time_series_dataset(
            target_name=self._prediction_name,
            time_name=input_data.time.name,
            extra_names=input_data.extras.column_names,
        )
