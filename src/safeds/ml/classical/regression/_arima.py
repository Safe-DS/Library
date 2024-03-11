from statsmodels.tsa.arima.model import ARIMA

import io
from typing import TYPE_CHECKING
import itertools
import matplotlib.pyplot as plt

from safeds.data.tabular.containers import TimeSeries
from safeds.data.image.containers import Image

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin
    from safeds.data.tabular.containers import TimeSeries


class ArimaModel:
    """Auto Regressive Integrated Moving Average Model."""

    def __init__(self) -> None:
        # Internal state
        self._arima = None
        self._order = None

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
            try:
                # Create and fit an ARIMA model with the current parameters
                mod = ARIMA(time_series.target._data.values, order=param)
                result = mod.fit()

                # Compare the current model's AIC with the best AIC so far
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_param = param
                    best_model = result
            except Exception as e:
                # Skip the iteration if the model cannot be fitted with current parameters
                print('ARIMA{} - AIC: skipped due to an error: {}'.format(param, e))
                continue

        fitted_arima._order = best_param
        fitted_arima._arima = best_model
        return fitted_arima

    def predict(self, time_series: TimeSeries):
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
