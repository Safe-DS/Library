from statsmodels.tsa.arima.model import ARIMA

import io
from typing import TYPE_CHECKING
import itertools
import matplotlib.pyplot as plt

from safeds.data.tabular.containers import TimeSeries, Column
from safeds.data.image.containers import Image

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin
    from safeds.data.tabular.containers import TimeSeries
from safeds.exceptions import (
    ModelNotFittedError,
    PredictionError
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
        training_set : TimeSeries
            The training data containing the feature and target vectors.

        Returns
        -------
        fitted_arima : ArimaModel
            The fitted ARIMA Model.

        Raises
        ------
        LearningError
            If the training data contains invalid values or if the training failed.
        UntaggedTableError
            If the table is untagged.
        NonNumericColumnError
            If the training data contains non-numerical values.
        MissingValuesColumnError
            If the training data contains missing values.
        DatasetMissesDataError
            If the training data contains no rows.
        """
        fitted_arima = ArimaModel()
        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q))
        best_aic = float("inf")
        best_model = None
        best_param = None
        for param in pdq:
            # Create and fit an ARIMA model with the current parameters
            mod = ARIMA(time_series.target._data.values, order=param)
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

    def predict(self, time_series: TimeSeries):
        """
            Predict a target vector using a time series target column. The model has to be trained first.

            Parameters
            ----------
            time_series : TimeSeries
                The time series containing the target vector.

            Returns
            -------
            table : TimeSeries
                A timeseries containing the old time series vectors and the predicted target vector.

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
        try:
            forecast_results = self._arima.forecast(steps=n_steps)
        except any:
            raise PredictionError

        # create new TimeSeries
        result = time_series.add_column(Column(name="forecasted", data = forecast_results))
        return result

    def plot_predictions(self, time_series: TimeSeries)-> Image:
        """
        Plot the predictions of the given time series target
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
        test_data = time_series.target._data.values
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
