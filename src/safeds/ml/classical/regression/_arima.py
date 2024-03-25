import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

import io
from typing import TYPE_CHECKING
import itertools
import matplotlib.pyplot as plt

from safeds.data.tabular.containers import TimeSeries, Column, Table
from safeds.data.image.containers import Image

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin
from safeds.exceptions import (
    ModelNotFittedError,
    PredictionError,
    DatasetMissesDataError,
    NonNumericColumnError,
    MissingValuesColumnError,
    LearningError,
    NonTimeSeriesError,
)


class ArimaModel:
    """Auto Regressive Integrated Moving Average Model."""

    def __init__(self) -> None:
        # Internal state
        self._arima = None
        self._order = None
        self._fitted = False

    def fit(self, time_series: TimeSeries):
        """
        Create a copy of this ARIMA Model and fit it with the given training data.

        This ARIMA Model is not modified.

        Parameters
        ----------
        time_series : TimeSeries
            The time series containing the target column, which will be used.

        Returns
        -------
        fitted_arima : ArimaModel
            The fitted ARIMA Model.

        Raises
        ------
        LearningError
            If the training data contains invalid values or if the training failed.
        NonTimeSeriesError
            If the table is not a TimeSeries object.
        NonNumericColumnError
            If the training data contains non-numerical values.
        MissingValuesColumnError
            If the training data contains missing values.
        DatasetMissesDataError
            If the training data contains no rows.
        """
        if not isinstance(time_series, TimeSeries) and isinstance(time_series, Table):
            raise NonTimeSeriesError
        if time_series.number_of_rows == 0:
            raise DatasetMissesDataError
        if not time_series.target.type.is_numeric():
            raise NonNumericColumnError(
                time_series.target.name
            )
        if time_series.target.has_missing_values():
            raise MissingValuesColumnError(
                time_series.target.name,
                "You can use the Imputer to replace the missing values based on different strategies.\nIf you want to"
                "remove the missing values entirely you can use the method "
                "`TimeSeries.remove_rows_with_missing_values`.",
            )
        fitted_arima = ArimaModel()
        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q))
        best_aic = float("inf")
        best_model = None
        best_param = None
        for param in pdq:
            # Create and fit an ARIMA model with the current parameters
            mod = ARIMA(time_series.target._data.values, order=param)
            try:
                result = mod.fit()
            except ValueError as exception:
                raise LearningError(str(exception)) from exception

            # Compare the current model's AIC with the best AIC so far
            if result.aic < best_aic:
                best_aic = result.aic
                best_param = param
                best_model = result

        fitted_arima._order = best_param
        fitted_arima._arima = best_model
        fitted_arima._fitted = True
        return fitted_arima

    def predict(self, forecast_horizon: int):
        """
            Predict a target vector using a time series target column. The model has to be trained first.

            Parameters
            ----------
            forecast_horizon : TimeSeries
                The forecast horizon for the predicted value.

            Returns
            -------
            table : TimeSeries
                A timeseries containing the predicted target vector and a time dummy as time column.

            Raises
            ------
            ModelNotFittedError
                If the model has not been fitted yet.
            IndexError
                If the forecast horizon is not greater than zero.
            PredictionError
                If predicting with the given dataset failed.
        """
        #Validation
        if forecast_horizon <= 0:
            raise IndexError("forecast_horizon must be greater 0")
        if not self.is_fitted():
            raise ModelNotFittedError

        try:
            forecast_results = self._arima.forecast(steps=forecast_horizon)
        except any as exception:
            raise PredictionError(str(exception)) from exception
        target_column = Column(name="target", data=forecast_results)
        time_column = Column(name="time", data= pd.Series(range(0, forecast_horizon),  name="time"))
        # create new TimeSeries
        result = Table()
        result = result.add_column(target_column)
        result = result.add_column(time_column)
        return TimeSeries._from_table(result, time_name="time", target_name="target")

    def plot_predictions(self, time_series: TimeSeries) -> Image:
        """
        Plot the predictions of the given time series target.

        Parameters
        ----------
        time_series : TimeSeries
            The time series containing the target vector.

        Returns
        -------
        image : Image
            Plots predictions of the given time series to the given target Column

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted yet.
        PredictionError
                If predicting with the given dataset failed.

        """
        if not self.is_fitted():
            raise ModelNotFittedError
        test_data = time_series.target._data.to_numpy()
        n_steps = len(test_data)
        forecast_results = self._arima.forecast(steps=n_steps)

        fig = plt.figure()
        plt.plot(forecast_results, )
        plt.legend(["forecasted"])
        plt.plot(test_data)
        plt.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        plt.close()  # Prevents the figure from being displayed directly
        buffer.seek(0)
        return Image.from_bytes(buffer.read())

    def is_fitted(self) -> bool:
        """
        Check if the classifier is fitted.

        Returns
        -------
        is_fitted : bool
            Whether the regressor is fitted.
        """
        return self._fitted
