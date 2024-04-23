from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

from safeds.data.tabular.containers import TimeSeries
from safeds.ml.nn._input_conversion import _InputConversion


class InputConversionTimeSeries(_InputConversion[TimeSeries, TimeSeries]):
    """The input conversion for a neural network, defines the input parameters for the neural network."""

    def __init__(
        self,
        window_size: int,
        forecast_horizon: int,
        target_name: str,
        time_name: str,
        feature_names: list[str] | None = None,
    ) -> None:
        """
        Define the input parameters for the neural network in the input conversion.

        Parameters
        ----------
        window_size
            The size of the created windows
        forecast_horizon
            The forecast horizon defines the future lag of the predicted values
        feature_names
            The names of the features for the input table, used as features for the training.
        target_name
            The name of the target for the input table, used as target for the training.
        """
        if feature_names is None:
            feature_names = []
        self._window_size = window_size
        self._forecast_horizon = forecast_horizon
        self._target_name = target_name
        self._time_name = time_name
        self._feature_names = feature_names

    @property
    def _data_size(self) -> int:
        """
        Gives the size for the input of an internal layer.

        Returns
        -------
        The size of the input for the neural network

        """
        return (len(self._feature_names) + 1) * self._window_size

    def _data_conversion_fit(self, input_data: TimeSeries, batch_size: int) -> DataLoader:
        return input_data._into_dataloader_with_window(
            self._window_size,
            self._forecast_horizon,
            batch_size,
        )

    def _data_conversion_predict(self, input_data: TimeSeries, batch_size: int) -> DataLoader:
        return input_data._into_dataloader_with_window_predict(self._window_size, self._forecast_horizon, batch_size)

    def _is_fit_data_valid(self, input_data: TimeSeries) -> bool:
        return (
            (sorted(input_data.features.column_names)).__eq__(sorted(self._feature_names))
            and input_data.target.name == self._target_name
            and input_data.time.name == self._time_name
        )

    def _is_predict_data_valid(self, input_data: TimeSeries) -> bool:
        return self._is_fit_data_valid(input_data)
