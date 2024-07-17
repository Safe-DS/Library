import copy
import multiprocessing as mp
from concurrent.futures import ALL_COMPLETED, wait
from typing import Self

from safeds._validation._check_columns_are_numeric import _check_columns_are_numeric
from safeds.data.labeled.containers import TabularDataset
from safeds.exceptions import (
    DatasetMissesDataError,
    FeatureDataMismatchError,
    ModelNotFittedError,
    TargetDataMismatchError,
)
from safeds.ml.classical.regression import (
    AdaBoostRegressor,
    DecisionTreeRegressor,
    ElasticNetRegressor,
    GradientBoostingRegressor,
    LassoRegressor,
    LinearRegressor,
    RandomForestRegressor,
    Regressor,
    RidgeRegressor,
    SupportVectorRegressor,
)


def _fit_single_model(model: Regressor, train_data: TabularDataset) -> Regressor:
    return model.fit(train_data)  # pragma: no cover


def _predict_single_model(model: Regressor, test_data: TabularDataset) -> TabularDataset:
    return model.predict(test_data)  # pragma: no cover


class BaselineRegressor:
    """
    Baseline Regressor.

    Get a baseline by fitting data on multiple different models and comparing the best metrics.

    Parameters
    ----------
    extended_search:
        If set to true, an extended set of models will be used to fit the classifier.
        This might result in significantly higher runtime.
    """

    def __init__(self, extended_search: bool = False):
        self._is_fitted = False
        self._list_of_model_types = [
            AdaBoostRegressor(),
            DecisionTreeRegressor(),
            LinearRegressor(),
            RandomForestRegressor(),
            RidgeRegressor(),
            SupportVectorRegressor(),
        ]

        if extended_search:
            self._list_of_model_types.extend(
                [ElasticNetRegressor(), LassoRegressor(), GradientBoostingRegressor()],
            )  # pragma: no cover

        self._fitted_models: list[Regressor] = []
        self._feature_names: list[str] | None = None
        self._target_name: str = "none"

    def fit(self, train_data: TabularDataset) -> Self:
        """
        Train the Regressor with given training data.

        The original model is not modified.

        Parameters
        ----------
        train_data:
            The data the network should be trained on.

        Returns
        -------
        trained_classifier:
            The trained Regressor

        Raises
        ------
        DatasetMissesDataError
            If the given train_data contains no data.
        ColumnTypeError
            If one or more columns contain non-numeric values.
        """
        from concurrent.futures import ProcessPoolExecutor

        # Validate Data
        train_data_as_table = train_data.to_table()
        if train_data_as_table.row_count == 0:
            raise DatasetMissesDataError
        _check_columns_are_numeric(train_data_as_table, train_data.features.add_columns(train_data.target).column_names)

        copied_model = copy.deepcopy(self)

        with ProcessPoolExecutor(
            max_workers=len(self._list_of_model_types),
            mp_context=mp.get_context("spawn"),
        ) as executor:
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
        """
        Make a prediction for the given test data and calculate the best metrics.

        The original Model is not modified.

        Parameters
        ----------
        test_data:
            The data the Regressor should predict.

        Returns
        -------
        best_metrics:
            A dictionary with the best metrics that were achieved.

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted yet
        FeatureDataMismatchError
            If the features of the test data do not match with the features of the trained Regressor.
        DatasetMissesDataError
            If the given test_data contains no data.
        TargetDataMismatchError
            If the target column of the test data does not match the target column of the training data.
        ColumnTypeError
            If one or more columns contain non-numeric values.
        """
        from concurrent.futures import ProcessPoolExecutor

        from safeds.ml.metrics import RegressionMetrics

        if not self._is_fitted:
            raise ModelNotFittedError

        # Validate data
        if not self._feature_names == test_data.features.column_names:
            raise FeatureDataMismatchError
        if not self._target_name == test_data.target.name:
            raise TargetDataMismatchError(
                actual_target_name=test_data.target.name,
                missing_target_name=self._target_name,
            )
        test_data_as_table = test_data.to_table()
        if test_data_as_table.row_count == 0:
            raise DatasetMissesDataError
        _check_columns_are_numeric(test_data_as_table, test_data.features.add_columns(test_data.target).column_names)

        # Start Processes
        with ProcessPoolExecutor(
            max_workers=len(self._list_of_model_types),
            mp_context=mp.get_context("spawn"),
        ) as executor:
            results = []
            futures = []
            for model in self._fitted_models:
                futures.append(executor.submit(_predict_single_model, model, test_data))
            [done, _] = wait(futures, return_when=ALL_COMPLETED)
            for future in done:
                results.append(future.result())
        executor.shutdown()

        # Calculate Metrics
        max_metrics = {
            "coefficient_of_determination": float("-inf"),
            "mean_absolute_error": float("inf"),
            "mean_squared_error": float("inf"),
            "median_absolute_deviation": float("inf"),
        }
        for result in results:
            coefficient_of_determination = RegressionMetrics.coefficient_of_determination(result, test_data)
            mean_absolute_error = RegressionMetrics.mean_absolute_error(result, test_data)
            mean_squared_error = RegressionMetrics.mean_squared_error(result, test_data)
            median_absolute_deviation = RegressionMetrics.median_absolute_deviation(result, test_data)

            if max_metrics.get("coefficient_of_determination", float("-inf")) < coefficient_of_determination:
                max_metrics.update({"coefficient_of_determination": coefficient_of_determination})

            if max_metrics.get("mean_absolute_error", float("inf")) > mean_absolute_error:
                max_metrics.update({"mean_absolute_error": mean_absolute_error})

            if max_metrics.get("mean_squared_error", float("inf")) > mean_squared_error:
                max_metrics.update({"mean_squared_error": mean_squared_error})

            if max_metrics.get("median_absolute_deviation", float("inf")) > median_absolute_deviation:
                max_metrics.update({"median_absolute_deviation": median_absolute_deviation})

        return max_metrics

    @property
    def is_fitted(self) -> bool:
        """Whether the model is fitted."""
        return self._is_fitted
