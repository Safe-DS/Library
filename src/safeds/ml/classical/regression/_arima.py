from __future__ import annotations

import io
import itertools

import matplotlib.pyplot as plt
import xxhash
from statsmodels.tsa.arima.model import ARIMA

from safeds.data.image.containers import Image
from safeds.data.tabular.containers import Column, Table, TimeSeries
from safeds.exceptions import (
    DatasetMissesDataError,
    MissingValuesColumnError,
    ModelNotFittedError,
    NonNumericColumnError,
    NonTimeSeriesError,
)


class ArimaModelRegressor:
    """Auto Regressive Integrated Moving Average Model."""

    def __hash__(self) -> int:
        """
        Return a deterministic hash value for a regressor.

        Returns
        -------
        hash : int
            The hash value.
        """
        return xxhash.xxh3_64(
            self.__class__.__qualname__.encode("utf-8")
            + (1 if self.is_fitted() else 0).to_bytes(1)
            + (bytes((9, 9, 9)) if self._order is None else bytes(self._order)),
        ).intdigest()

    def __init__(self) -> None:
        # Internal state
        self._arima: ARIMA | None = None
        self._order: tuple[int, int, int] | None = None
        self._fitted = False

    def fit(self, time_series: TimeSeries) -> ArimaModelRegressor:
        """
        Create a copy of this ARIMA Model and fit it with the given training data.

        This ARIMA Model is not modified.

        Parameters
        ----------
        time_series:
            The time series containing the target column, which will be used.

        Returns
        -------
        fitted_arima:
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
            raise NonNumericColumnError(time_series.target.name)
        if time_series.target.has_missing_values():
            raise MissingValuesColumnError(
                time_series.target.name,
                "You can use the Imputer to replace the missing values based on different strategies.\nIf you want to"
                "remove the missing values entirely you can use the method "
                "`TimeSeries.remove_rows_with_missing_values`.",
            )
        fitted_arima = ArimaModelRegressor()
        p = d = q = range(2)
        pdq = list(itertools.product(p, d, q))
        best_aic = float("inf")
        best_model = None
        # best param will get overwritten
        best_param = (0, 0, 0)
        for param in pdq:
            # Create and fit an ARIMA model with the current parameters
            mod = ARIMA(time_series.target._data.values, order=param)

            # I wasnt able to invoke an learning Error
            # Add try catch when an learning error is found
            result = mod.fit()

            # Compare the current model's AIC with the best AIC so far
            if result.aic < best_aic:
                best_aic = result.aic
                best_param = param
                best_model = result

        fitted_arima._order = best_param
        fitted_arima._arima = best_model
        fitted_arima._fitted = True
        return fitted_arima

    def predict(self, time_series: TimeSeries) -> TimeSeries:
        """
        Predict a target vector using a time series target column. The model has to be trained first.

        Parameters
        ----------
        time_series:
            The test dataset of the time series.

        Returns
        -------
        table:
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
        # make a table without
        forecast_horizon = len(time_series.target._data)
        result_table = time_series._as_table()
        result_table = result_table.remove_columns([time_series.target.name])
        # Validation
        if not self.is_fitted() or self._arima is None:
            raise ModelNotFittedError

        # forecast
        # couldn't invoke prediction error, will be added when found
        forecast_results = self._arima.forecast(steps=forecast_horizon)
        target_column: Column = Column(name=time_series.target.name + " " + "forecasted", data=forecast_results)

        # create new TimeSeries
        result_table = result_table.add_column(target_column)
        return TimeSeries._from_table(
            result_table,
            time_name=time_series.time.name,
            target_name=time_series.target.name + " " + "forecasted",
            feature_names=time_series.features.column_names,
        )

    def plot_predictions(self, test_series: TimeSeries) -> Image:
        """
        Plot the predictions of the trained model to the given target of the time series. So you can see the predictions and the actual values in one plot.

        Parameters
        ----------
        test_series:
            The time series containing the target vector.

        Returns
        -------
        image:
            Plots predictions of the given time series to the given target Column

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted yet.
        PredictionError
            If predicting with the given dataset failed.

        """
        if not self.is_fitted() or self._arima is None:
            raise ModelNotFittedError
        test_data = test_series.target._data.to_numpy()
        n_steps = len(test_data)
        forecast_results = self._arima.forecast(steps=n_steps)

        fig = plt.figure()
        plt.plot(
            forecast_results,
        )
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
        is_fitted:
            Whether the regressor is fitted.
        """
        return self._fitted
