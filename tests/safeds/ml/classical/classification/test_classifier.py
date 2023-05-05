from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.ml.classical.classification import (
    AdaBoost,
    Classifier,
    DecisionTree,
    GradientBoosting,
    KNearestNeighbors,
    LogisticRegression,
    RandomForest,
    SupportVectorMachine,
)
from safeds.ml.exceptions import (
    DatasetContainsTargetError,
    DatasetMissesFeaturesError,
    LearningError,
    ModelNotFittedError,
    PredictionError,
    UntaggedTableError,
)

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin
    from _pytest.fixtures import FixtureRequest


def classifiers() -> list[Classifier]:
    """
    Return the list of classifiers to test.

    After you implemented a new classifier, add it to this list to ensure its `fit` and `predict` method work as
    expected. Place tests of methods that are specific to your classifier in a separate test file.

    Returns
    -------
    classifiers : list[Classifier]
        The list of classifiers to test.
    """
    return [
        AdaBoost(),
        DecisionTree(),
        GradientBoosting(),
        KNearestNeighbors(2),
        LogisticRegression(),
        RandomForest(),
        SupportVectorMachine(),
    ]


@pytest.fixture()
def valid_data() -> TaggedTable:
    return Table.from_dict(
        {
            "id": [1, 4],
            "feat1": [2, 5],
            "feat2": [3, 6],
            "target": [0, 1],
        },
    ).tag_columns(target_name="target", feature_names=["feat1", "feat2"])


@pytest.fixture()
def invalid_data() -> TaggedTable:
    return Table.from_dict(
        {
            "id": [1, 4],
            "feat1": ["a", 5],
            "feat2": [3, 6],
            "target": [0, 1],
        },
    ).tag_columns(target_name="target", feature_names=["feat1", "feat2"])


@pytest.mark.parametrize("classifier", classifiers(), ids=lambda x: x.__class__.__name__)
class TestFit:
    def test_should_succeed_on_valid_data(self, classifier: Classifier, valid_data: TaggedTable) -> None:
        classifier.fit(valid_data)
        assert True  # This asserts that the fit method succeeds

    def test_should_not_change_input_classifier(self, classifier: Classifier, valid_data: TaggedTable) -> None:
        classifier.fit(valid_data)
        assert not classifier.is_fitted()

    def test_should_not_change_input_table(self, classifier: Classifier, request: FixtureRequest) -> None:
        valid_data = request.getfixturevalue("valid_data")
        valid_data_copy = request.getfixturevalue("valid_data")
        classifier.fit(valid_data)
        assert valid_data == valid_data_copy

    def test_should_raise_on_invalid_data(self, classifier: Classifier, invalid_data: TaggedTable) -> None:
        with pytest.raises(LearningError):
            classifier.fit(invalid_data)

    @pytest.mark.parametrize(
        "table",
        [
            Table.from_dict(
                {
                    "a": [1.0, 0.0, 0.0, 0.0],
                    "b": [0.0, 1.0, 1.0, 0.0],
                    "c": [0.0, 0.0, 0.0, 1.0],
                },
            ),
        ],
        ids=["untagged_table"],
    )
    def test_should_raise_if_table_is_not_tagged(self, classifier: Classifier, table: Table) -> None:
        with pytest.raises(UntaggedTableError):
            classifier.fit(table)  # type: ignore[arg-type]


@pytest.mark.parametrize("classifier", classifiers(), ids=lambda x: x.__class__.__name__)
class TestPredict:
    def test_should_include_features_of_input_table(self, classifier: Classifier, valid_data: TaggedTable) -> None:
        fitted_classifier = classifier.fit(valid_data)
        prediction = fitted_classifier.predict(valid_data.features)
        assert prediction.features == valid_data.features

    def test_should_include_complete_input_table(self, classifier: Classifier, valid_data: TaggedTable) -> None:
        fitted_regressor = classifier.fit(valid_data)
        prediction = fitted_regressor.predict(valid_data.remove_columns(["target"]))
        assert prediction.remove_columns(["target"]) == valid_data.remove_columns(["target"])

    def test_should_set_correct_target_name(self, classifier: Classifier, valid_data: TaggedTable) -> None:
        fitted_classifier = classifier.fit(valid_data)
        prediction = fitted_classifier.predict(valid_data.features)
        assert prediction.target.name == "target"

    def test_should_not_change_input_table(self, classifier: Classifier, request: FixtureRequest) -> None:
        valid_data = request.getfixturevalue("valid_data")
        valid_data_copy = request.getfixturevalue("valid_data")
        fitted_classifier = classifier.fit(valid_data)
        fitted_classifier.predict(valid_data.features)
        assert valid_data == valid_data_copy

    def test_should_raise_if_not_fitted(self, classifier: Classifier, valid_data: TaggedTable) -> None:
        with pytest.raises(ModelNotFittedError):
            classifier.predict(valid_data.features)

    def test_should_raise_if_dataset_contains_target(self, classifier: Classifier, valid_data: TaggedTable) -> None:
        fitted_classifier = classifier.fit(valid_data)
        with pytest.raises(DatasetContainsTargetError, match="target"):
            fitted_classifier.predict(valid_data)

    def test_should_raise_if_dataset_misses_features(self, classifier: Classifier, valid_data: TaggedTable) -> None:
        fitted_classifier = classifier.fit(valid_data)
        with pytest.raises(DatasetMissesFeaturesError, match="[feat1, feat2]"):
            fitted_classifier.predict(valid_data.remove_columns(["feat1", "feat2", "target"]))

    def test_should_raise_on_invalid_data(
        self,
        classifier: Classifier,
        valid_data: TaggedTable,
        invalid_data: TaggedTable,
    ) -> None:
        fitted_classifier = classifier.fit(valid_data)
        with pytest.raises(PredictionError):
            fitted_classifier.predict(invalid_data.features)


@pytest.mark.parametrize("classifier", classifiers(), ids=lambda x: x.__class__.__name__)
class TestIsFitted:
    def test_should_return_false_before_fitting(self, classifier: Classifier) -> None:
        assert not classifier.is_fitted()

    def test_should_return_true_after_fitting(self, classifier: Classifier, valid_data: TaggedTable) -> None:
        fitted_classifier = classifier.fit(valid_data)
        assert fitted_classifier.is_fitted()


class DummyClassifier(Classifier):
    """
    Dummy classifier to test metrics.

    Metrics methods expect a `TaggedTable` as input with two columns:

    - `predicted`: The predicted targets.
    - `expected`: The correct targets.

    `target_name` must be set to `"expected"`.
    """

    def fit(self, training_set: TaggedTable) -> DummyClassifier:  # noqa: ARG002
        return self

    def predict(self, dataset: Table) -> TaggedTable:
        # Needed until https://github.com/Safe-DS/Stdlib/issues/75 is fixed
        predicted = dataset.get_column("predicted")
        feature = predicted.rename("feature")
        dataset = Table.from_columns([feature, predicted])

        return dataset.tag_columns(target_name="predicted")

    def is_fitted(self) -> bool:
        return True

    def _get_sklearn_classifier(self) -> ClassifierMixin:
        pass


class TestAccuracy:
    def test_with_same_type(self) -> None:
        table = Table.from_dict(
            {
                "predicted": [1, 2, 3, 4],
                "expected": [1, 2, 3, 3],
            },
        ).tag_columns(target_name="expected")

        assert DummyClassifier().accuracy(table) == 0.75

    def test_with_different_types(self) -> None:
        table = Table.from_dict(
            {
                "predicted": ["1", "2", "3", "4"],
                "expected": [1, 2, 3, 3],
            },
        ).tag_columns(target_name="expected")

        assert DummyClassifier().accuracy(table) == 0.0

    @pytest.mark.parametrize(
        "table",
        [
            Table.from_dict(
                {
                    "a": [1.0, 0.0, 0.0, 0.0],
                    "b": [0.0, 1.0, 1.0, 0.0],
                    "c": [0.0, 0.0, 0.0, 1.0],
                },
            ),
        ],
        ids=["untagged_table"],
    )
    def test_should_raise_if_table_is_not_tagged(self, table: Table) -> None:
        with pytest.raises(UntaggedTableError):
            DummyClassifier().accuracy(table)  # type: ignore[arg-type]
