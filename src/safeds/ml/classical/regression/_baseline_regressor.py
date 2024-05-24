import copy
from typing import Self

from safeds.data.labeled.containers import TabularDataset
from safeds.exceptions import ModelNotFittedError
from safeds.ml.classical.regression import AdaBoostRegressor, DecisionTreeRegressor, ElasticNetRegressor, GradientBoostingRegressor, KNearestNeighborsRegressor, LassoRegressor, LinearRegressor, RandomForestRegressor, RidgeRegressor, SupportVectorRegressor
from safeds.ml.classical.regression import Regressor


def _fit_single_model(model: Regressor, train_data: TabularDataset) -> Regressor:
    return model.fit(train_data)


def _predict_single_model(model: Regressor, test_data: TabularDataset) -> TabularDataset:
    return model.predict(test_data)


class BaselineRegressor:
    def __init__(self):
        self._is_fitted = False
        self._list_of_model_types = [AdaBoostRegressor(), DecisionTreeRegressor(), ElasticNetRegressor(), GradientBoostingRegressor(), KNearestNeighborsRegressor(5), LassoRegressor(), LinearRegressor(), RandomForestRegressor(), RidgeRegressor(), SupportVectorRegressor()]
        self._fitted_models = []

    def fit(self, train_data: TabularDataset) -> Self:
        from concurrent.futures import ProcessPoolExecutor

        #Todo Validate data
        copied_model = copy.deepcopy(self)

        with ProcessPoolExecutor(max_workers=len(self._list_of_model_types)) as executor:
            futures = []
            for model in self._list_of_model_types:
                futures.append(executor.submit(_fit_single_model, model, train_data))
            for future in futures:
                copied_model._fitted_models.append(future.result())
        executor.shutdown()

        copied_model._is_fitted = True
        return copied_model

    def predict(self, test_data: TabularDataset) -> dict[str, float]:
        from concurrent.futures import ProcessPoolExecutor
        from safeds.ml.metrics import RegressionMetrics

        if not self._is_fitted:
            raise ModelNotFittedError

        #Todo Validate data

        with ProcessPoolExecutor(max_workers=len(self._list_of_model_types)) as executor:
            results = []
            futures = []
            for model in self._fitted_models:
                futures.append(executor.submit(_predict_single_model, model, test_data))
            for future in futures:
                results.append(future.result())
        executor.shutdown()

        max_metrics = {"coefficient_of_determination": 10000000000000000000000.0, "mean_absolute_error": 10000000000000000000000.0, "mean_squared_error": 10000000000000000000000.0, "median_absolute_deviation": 10000000000000000000000.0}
        for result in results:
            coefficient_of_determination = RegressionMetrics.coefficient_of_determination(result, test_data)
            mean_absolute_error = RegressionMetrics.mean_squared_error(result, test_data)
            mean_squared_error = RegressionMetrics.mean_absolute_error(result, test_data)
            median_absolute_deviation = RegressionMetrics.median_absolute_deviation(result, test_data)

            if max_metrics.get("coefficient_of_determination") > coefficient_of_determination:
                max_metrics.update({"coefficient_of_determination": coefficient_of_determination})

            if max_metrics.get("mean_absolute_error") > mean_absolute_error:
                max_metrics.update({"mean_absolute_error": mean_absolute_error})

            if max_metrics.get("mean_squared_error") > mean_squared_error:
                max_metrics.update({"mean_squared_error": mean_squared_error})

            if max_metrics.get("median_absolute_deviation") > median_absolute_deviation:
                max_metrics.update({"median_absolute_deviation": median_absolute_deviation})

        print(max_metrics)
        return max_metrics

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
