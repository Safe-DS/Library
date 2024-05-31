import copy
import time
from concurrent.futures import as_completed, FIRST_COMPLETED, wait, ALL_COMPLETED
from typing import Self

from safeds._validation._check_columns_are_numeric import _check_columns_are_numeric
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import ModelNotFittedError, NonNumericColumnError, DatasetMissesDataError, \
    FeatureDataMismatchError
from safeds.ml.classical.regression import AdaBoostRegressor, DecisionTreeRegressor, ElasticNetRegressor, \
    GradientBoostingRegressor, KNearestNeighborsRegressor, LassoRegressor, LinearRegressor, RandomForestRegressor, \
    RidgeRegressor, SupportVectorRegressor
from safeds.ml.classical.regression import Regressor


def _fit_single_model(model: Regressor, train_data: TabularDataset) -> Regressor:
    return model.fit(train_data)


def _predict_single_model(model: Regressor, test_data: TabularDataset) -> TabularDataset:
    return model.predict(test_data)


class BaselineRegressor:
    def __init__(self, include_slower_models: bool = False):
        self._is_fitted = False
        #TODO maybe add KNearestNeighbors
        self._list_of_model_types = [AdaBoostRegressor(), DecisionTreeRegressor(),
                                     LinearRegressor(), RandomForestRegressor(), RidgeRegressor(),
                                     SupportVectorRegressor()]

        if include_slower_models:
            self._list_of_model_types.extend([ElasticNetRegressor(), LassoRegressor(), GradientBoostingRegressor()])

        self._fitted_models = []
        self._feature_names = None
        self._target_name = None

    def fit(self, train_data: TabularDataset) -> Self:
        from concurrent.futures import ProcessPoolExecutor

        #Validate Data
        train_data_as_table = train_data.to_table()
        if train_data_as_table.row_count == 0:
            raise DatasetMissesDataError
        _check_columns_are_numeric(train_data_as_table, train_data.features.add_columns(train_data.target).column_names)

        copied_model = copy.deepcopy(self)

        with ProcessPoolExecutor(max_workers=len(self._list_of_model_types)) as executor:
            futures = []
            for model in self._list_of_model_types:
                futures.append(executor.submit(_fit_single_model, model, train_data))
            [done, _] = wait(futures, return_when=ALL_COMPLETED)
            for future in done:
                copied_model._fitted_models.append(future.result())
        executor.shutdown()

        copied_model._is_fitted = True
        copied_model._feature_names = train_data.features.column_names
        copied_model._target_name = train_data.target.name
        return copied_model

    def predict(self, test_data: TabularDataset) -> dict[str, float]:
        #TODO Think about combining fit and predict into one method
        from concurrent.futures import ProcessPoolExecutor
        from safeds.ml.metrics import RegressionMetrics

        if not self._is_fitted:
            raise ModelNotFittedError

        # Validate data
        if not self._feature_names == test_data.features.column_names:
            raise FeatureDataMismatchError
        #if not self._target_name == test_data.target.name:
        #    raise TODO Create new Error for this Case?
        test_data_as_table = test_data.to_table()
        if test_data_as_table.row_count == 0:
            raise DatasetMissesDataError
        _check_columns_are_numeric(test_data_as_table, test_data.features.add_columns(test_data.target).column_names)

        # Start Processes
        with ProcessPoolExecutor(max_workers=len(self._list_of_model_types)) as executor:
            results = []
            futures = []
            for model in self._fitted_models:
                futures.append(executor.submit(_predict_single_model, model, test_data))
            [done, _] = wait(futures, return_when=ALL_COMPLETED)
            for future in done:
                results.append(future.result())
        executor.shutdown()

        # Calculate Metrics
        max_metrics = {"coefficient_of_determination": float('-inf'), "mean_absolute_error": float('inf'),
                       "mean_squared_error": float('inf'), "median_absolute_deviation": float('inf')}
        for result in results:
            coefficient_of_determination = RegressionMetrics.coefficient_of_determination(result, test_data)
            mean_absolute_error = RegressionMetrics.mean_absolute_error(result, test_data)
            mean_squared_error = RegressionMetrics.mean_squared_error(result, test_data)
            median_absolute_deviation = RegressionMetrics.median_absolute_deviation(result, test_data)

            if max_metrics.get("coefficient_of_determination") < coefficient_of_determination:
                max_metrics.update({"coefficient_of_determination": coefficient_of_determination})

            if max_metrics.get("mean_absolute_error") > mean_absolute_error:
                max_metrics.update({"mean_absolute_error": mean_absolute_error})

            if max_metrics.get("mean_squared_error") > mean_squared_error:
                max_metrics.update({"mean_squared_error": mean_squared_error})

            if max_metrics.get("median_absolute_deviation") > median_absolute_deviation:
                max_metrics.update({"median_absolute_deviation": median_absolute_deviation})

        print(Table(
            {
                "Metric": [
                    "coefficient_of_determination",
                    "mean_absolute_error",
                    "mean_squared_error",
                    "median_absolute_deviation",
                ],
                "Best value": [
                    max_metrics.get("coefficient_of_determination"),
                    max_metrics.get("mean_absolute_error"),
                    max_metrics.get("mean_squared_error"),
                    max_metrics.get("median_absolute_deviation"),
                ],
            },
        ))
        return max_metrics

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
