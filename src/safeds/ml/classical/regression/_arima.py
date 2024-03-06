from statsmodels.tsa.arima.model import ARIMA

import io
from typing import TYPE_CHECKING
import itertools
import matplotlib.pyplot as plt

from safeds.data.tabular.containers import TimeSeries
from safeds.data.image.containers import Image

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin
    from safeds.data.tabular.containers import Table, TaggedTable, TimeSeries


class arimaModel():
    """Auto Regressive Integrated Moving Average Model."""

    def __init__(self) -> None:
        # Internal state
        self._arima = None
        self._order = None

    def fit(self, time_series: TimeSeries):
        result = arimaModel()
        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q))
        best_aic = float("inf")
        best_model = None
        best_param = None
        for param in pdq:
            try:
                # Create and fit an ARIMA model with the current parameters
                size = len(time_series.target._data.values)
                mod = ARIMA(time_series.target._data.values[:int(size*0.8)], order=param)
                result = mod.fit()

                # Compare the current model's AIC with the best AIC so far
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_model = result
                    print(param)
                    best_param = param
                print('ARIMA{} AIC:{}'.format(param, result.aic))
            except Exception as e:
                # Skip the iteration if the model cannot be fitted with current parameters
                print('ARIMA{} - AIC: skipped due to an error: {}'.format(param, e))
                continue

        self._order = best_param
        return result

    def predict(self, time_series: TimeSeries):
        print(self._order)
        size = len(time_series.target._data.values)
        temp_arima =ARIMA(time_series.target._data.values[:int(size*0.8)], order=self._order)
        temp_fitted = temp_arima.fit()

        test_data = time_series.target._data.values[int(size*0.8):]
        n_steps = len(test_data)
        print(n_steps)
        forecast_results = temp_fitted.forecast(steps=n_steps)

        print(len(forecast_results))
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
