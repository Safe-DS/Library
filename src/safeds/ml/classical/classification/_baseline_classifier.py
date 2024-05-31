import copy
from concurrent.futures import wait, ALL_COMPLETED, ProcessPoolExecutor
from typing import Self

from safeds._validation._check_columns_are_numeric import _check_columns_are_numeric
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import ModelNotFittedError, DatasetMissesDataError, FeatureDataMismatchError
from safeds.ml.classical.classification import Classifier
from safeds.ml.classical.classification import RandomForestClassifier, AdaBoostClassifier, \
            DecisionTreeClassifier, GradientBoostingClassifier, KNearestNeighborsClassifier, SupportVectorClassifier


def _fit_single_model(model: Classifier, train_data: TabularDataset) -> Classifier:
    return model.fit(train_data)


def _predict_single_model(model: Classifier, test_data: TabularDataset) -> TabularDataset:
    return model.predict(test_data)


class BaselineClassifier:
    def __init__(self, include_slower_models: bool = False):
        self._is_fitted = False
        self._list_of_model_types = [AdaBoostClassifier(), DecisionTreeClassifier(),
                          SupportVectorClassifier(), RandomForestClassifier()]
        # TODO maybe add KNearestNeighbors to extended models
        if include_slower_models:
            self._list_of_model_types.extend([GradientBoostingClassifier()])

        self._fitted_models = []
        self._feature_names = None
        self._target_name = None

    def fit(self, train_data: TabularDataset) -> Self:
        from concurrent.futures import ProcessPoolExecutor

        # Validate Data
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
            for future in futures:
                copied_model._fitted_models.append(future.result())
        executor.shutdown()

        copied_model._is_fitted = True
        copied_model._feature_names = train_data.features.column_names
        copied_model._target_name = train_data.target.name
        return copied_model

    def predict(self, test_data: TabularDataset) -> dict[str, float]:
        # TODO Think about combining fit and predict into one method
        from concurrent.futures import ProcessPoolExecutor
        from safeds.ml.metrics import ClassificationMetrics

        if not self._is_fitted:
            raise ModelNotFittedError

        # Validate data
        if not self._feature_names == test_data.features.column_names:
            raise FeatureDataMismatchError
        # if not self._target_name == test_data.target.name:
        #    raise TODO Create new Error for this Case?
        test_data_as_table = test_data.to_table()
        if test_data_as_table.row_count == 0:
            raise DatasetMissesDataError
        _check_columns_are_numeric(test_data_as_table, test_data.features.add_columns(test_data.target).column_names)

        with ProcessPoolExecutor(max_workers=len(self._list_of_model_types)) as executor:
            results = []
            futures = []
            for model in self._fitted_models:
                futures.append(executor.submit(_predict_single_model, model, test_data))
            for future in futures:
                results.append(future.result())
        executor.shutdown()

        max_metrics = {"accuracy": 0.0, "f1score": 0.0, "precision": 0.0, "recall": 0.0}
        for result in results:
            accuracy = ClassificationMetrics.accuracy(result, test_data)

            positive_class = test_data.target.get_value(0)
            f1score = ClassificationMetrics.f1_score(result, test_data, positive_class)
            precision = ClassificationMetrics.precision(result, test_data, positive_class)
            recall = ClassificationMetrics.recall(result, test_data, positive_class)

            if max_metrics.get("accuracy") < accuracy:
                max_metrics.update({"accuracy": accuracy})

            if max_metrics.get("f1score") < f1score:
                max_metrics.update({"f1score": f1score})

            if max_metrics.get("precision") < precision:
                max_metrics.update({"precision": precision})

            if max_metrics.get("recall") < recall:
                max_metrics.update({"recall": recall})

        print(Table(
            {
                "Metric": [
                    "accuracy",
                    "f1score",
                    "precision",
                    "recall",
                ],
                "Best value": [
                    max_metrics.get("accuracy"),
                    max_metrics.get("f1score"),
                    max_metrics.get("precision"),
                    max_metrics.get("recall"),
                ],
            },
        ))
        return max_metrics

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
