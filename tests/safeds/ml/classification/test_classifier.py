from __future__ import annotations

import pandas as pd
import pytest
from _pytest.fixtures import FixtureRequest

from safeds.data.tabular.containers import Column, Table, TaggedTable
from safeds.exceptions import LearningError, PredictionError
from safeds.ml.classification import Classifier, AdaBoost, DecisionTree, GradientBoosting, KNearestNeighbors, \
    LogisticRegression, RandomForest


def classifiers() -> list[Classifier]:
    """
    Returns the list of classifiers to test.

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
        RandomForest()
    ]


@pytest.fixture()
def valid_data() -> TaggedTable:
    return Table.from_columns(
        [
            Column("id", [1, 4]),
            Column("feat1", [2, 5]),
            Column("feat2", [3, 6]),
            Column("target", [0, 1]),
        ]
    ).tag_columns(target_name="target", feature_names=["feat1", "feat2"])


@pytest.fixture()
def invalid_data() -> TaggedTable:
    return Table.from_columns(
        [
            Column("id", [1, 4]),
            Column("feat1", ["a", 5]),
            Column("feat2", [3, 6]),
            Column("target", [0, 1]),
        ]
    ).tag_columns(target_name="target", feature_names=["feat1", "feat2"])


@pytest.mark.parametrize(
    "classifier",
    classifiers(),
    ids=lambda x: x.__class__.__name__
)
class TestFit:
    def test_should_succeed_on_valid_data(self, classifier: Classifier, valid_data: TaggedTable) -> None:
        classifier.fit(valid_data)
        assert True  # This asserts that the fit method succeeds

    def test_should_not_change_input_table(self, classifier: Classifier, request: FixtureRequest) -> None:
        valid_data = request.getfixturevalue("valid_data")
        valid_data_copy = request.getfixturevalue("valid_data")
        classifier.fit(valid_data)
        assert valid_data == valid_data_copy

    def test_should_raise_on_invalid_data(self, classifier: Classifier, invalid_data: TaggedTable) -> None:
        with pytest.raises(LearningError):
            classifier.fit(invalid_data)


@pytest.mark.parametrize(
    "classifier",
    classifiers(),
    ids=lambda x: x.__class__.__name__
)
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

    def test_should_raise_when_not_fitted(self, classifier: Classifier, valid_data: TaggedTable) -> None:
        with pytest.raises(PredictionError):
            classifier.predict(valid_data.features)

    def test_should_raise_on_invalid_data(
        self, classifier: Classifier, valid_data: TaggedTable, invalid_data: TaggedTable
    ) -> None:
        fitted_classifier = classifier.fit(valid_data)
        with pytest.raises(PredictionError):
            fitted_classifier.predict(invalid_data.features)


class DummyClassifier(Classifier):
    """
    Dummy classifier to test metrics.

    Metrics methods expect a `TaggedTable` as input with two columns:

    - `predicted`: The predicted targets.
    - `expected`: The correct targets.

    `target_name` must be set to `"expected"`.
    """

    def fit(self, training_set: TaggedTable) -> DummyClassifier:
        # pylint: disable=unused-argument
        return self

    def predict(self, dataset: Table) -> TaggedTable:
        # Needed until https://github.com/Safe-DS/Stdlib/issues/75 is fixed
        predicted = dataset.get_column("predicted")
        feature = predicted.rename("feature")
        dataset = Table.from_columns([feature, predicted])

        return dataset.tag_columns(target_name="predicted")


class TestAccuracy:
    def test_with_same_type(self) -> None:
        c1 = Column("predicted", [1, 2, 3, 4])
        c2 = Column("expected", [1, 2, 3, 3])
        table = Table.from_columns([c1, c2]).tag_columns(target_name="expected")

        assert DummyClassifier().accuracy(table) == 0.75

    def test_with_different_types(self) -> None:
        c1 = Column("predicted", pd.Series(data=["1", "2", "3", "4"]))
        c2 = Column("expected", pd.Series(data=[1, 2, 3, 3]))
        table = Table.from_columns([c1, c2]).tag_columns(target_name="expected")

        assert DummyClassifier().accuracy(table) == 0.0
