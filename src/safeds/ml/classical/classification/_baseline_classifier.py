import copy
from typing import Self

from safeds.data.labeled.containers import TabularDataset
from safeds.exceptions import ModelNotFittedError
from safeds.ml.classical.classification import Classifier
from safeds.ml.classical.classification import RandomForestClassifier, AdaBoostClassifier, \
            DecisionTreeClassifier, GradientBoostingClassifier, KNearestNeighborsClassifier, SupportVectorClassifier


def _fit_single_model(model: Classifier, train_data: TabularDataset) -> Classifier:
    return model.fit(train_data)


def _predict_single_model(model: Classifier, test_data: TabularDataset) -> TabularDataset:
    return model.predict(test_data)


class BaselineClassifier:
    def __init__(self):
        self._is_fitted = False
        self._list_of_model_types = [AdaBoostClassifier(), DecisionTreeClassifier(), GradientBoostingClassifier(),
                          SupportVectorClassifier(), KNearestNeighborsClassifier(2), RandomForestClassifier()]
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
        from safeds.ml.metrics import ClassificationMetrics

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

        print(max_metrics)
        return max_metrics

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
