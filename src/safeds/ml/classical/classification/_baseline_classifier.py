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
from safeds.ml.classical.classification import (
    AdaBoostClassifier,
    Classifier,
    DecisionTreeClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    SupportVectorClassifier,
)


def _fit_single_model(model: Classifier, train_data: TabularDataset) -> Classifier:
    return model.fit(train_data)  # pragma: no cover


def _predict_single_model(model: Classifier, test_data: TabularDataset) -> TabularDataset:
    return model.predict(test_data)  # pragma: no cover


class BaselineClassifier:
    """
    Baseline Classifier.

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
            AdaBoostClassifier(),
            DecisionTreeClassifier(),
            SupportVectorClassifier(),
            RandomForestClassifier(),
        ]
        if extended_search:
            self._list_of_model_types.extend([GradientBoostingClassifier()])  # pragma: no cover

        self._fitted_models: list[Classifier] = []
        self._feature_names: list[str] | None = None
        self._target_name: str = "none"

    def fit(self, train_data: TabularDataset) -> Self:
        """
        Train the Classifier with given training data.

        The original model is not modified.

        Parameters
        ----------
        train_data:
            The data the network should be trained on.

        Returns
        -------
        trained_classifier:
            The trained Classifier

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
            The data the Classifier should predict.

        Returns
        -------
        best_metrics:
            A dictionary with the best metrics that were achieved.

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted yet
        FeatureDataMismatchError
            If the features of the test data do not match with the features of the trained Classifier.
        DatasetMissesDataError
            If the given test_data contains no data.
        TargetDataMismatchError
            If the target column of the test data does not match the target column of the training data.
        ColumnTypeError
            If one or more columns contain non-numeric values.
        """
        from concurrent.futures import ProcessPoolExecutor

        from safeds.ml.metrics import ClassificationMetrics

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

        max_metrics = {"accuracy": 0.0, "f1score": 0.0, "precision": 0.0, "recall": 0.0}
        for result in results:
            accuracy = ClassificationMetrics.accuracy(result, test_data)

            positive_class = test_data.target.get_value(0)
            f1score = ClassificationMetrics.f1_score(result, test_data, positive_class)
            precision = ClassificationMetrics.precision(result, test_data, positive_class)
            recall = ClassificationMetrics.recall(result, test_data, positive_class)

            if max_metrics.get("accuracy", 0.0) < accuracy:
                max_metrics.update({"accuracy": accuracy})

            if max_metrics.get("f1score", 0.0) < f1score:
                max_metrics.update({"f1score": f1score})

            if max_metrics.get("precision", 0.0) < precision:
                max_metrics.update({"precision": precision})

            if max_metrics.get("recall", 0.0) < recall:
                max_metrics.update({"recall": recall})

        return max_metrics

    @property
    def is_fitted(self) -> bool:
        """Whether the model is fitted."""
        return self._is_fitted
