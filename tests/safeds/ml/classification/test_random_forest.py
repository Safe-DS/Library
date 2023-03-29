import pytest

from safeds.data.tabular.containers import Table, TaggedTable, Column
from safeds.exceptions import LearningError, PredictionError
from safeds.ml.classification import Classifier, RandomForest


@pytest.fixture()
def classifier() -> Classifier:
    return RandomForest()


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


class TestFit:
    def test_should_succeed_on_valid_data(self, classifier: Classifier, valid_data: TaggedTable) -> None:
        classifier.fit(valid_data)
        assert True  # This asserts that the fit method succeeds

    def test_should_raise_on_invalid_data(self, classifier: Classifier, invalid_data: TaggedTable) -> None:
        with pytest.raises(LearningError):
            classifier.fit(invalid_data)


class TestPredict:
    def test_should_include_features_of_prediction_input(self, classifier: Classifier, valid_data: TaggedTable) -> None:
        fitted_classifier = classifier.fit(valid_data)
        prediction = fitted_classifier.predict(valid_data.features)
        assert prediction.features == valid_data.features

    def test_should_include_complete_prediction_input(self, classifier: Classifier, valid_data: TaggedTable) -> None:
        fitted_regressor = classifier.fit(valid_data)
        prediction = fitted_regressor.predict(valid_data.drop_columns(["target"]))
        assert prediction.drop_columns(["target"]) == valid_data.drop_columns(["target"])

    def test_should_set_correct_target_name(self, classifier: Classifier, valid_data: TaggedTable) -> None:
        fitted_classifier = classifier.fit(valid_data)
        prediction = fitted_classifier.predict(valid_data.features)
        assert prediction.target.name == "target"

    def test_should_raise_when_not_fitted(self, classifier: Classifier, valid_data: TaggedTable) -> None:
        with pytest.raises(PredictionError):
            classifier.predict(valid_data.features)

    def test_should_raise_on_invalid_data(
        self, classifier: Classifier, valid_data: TaggedTable, invalid_data: TaggedTable
    ) -> None:
        fitted_classifier = classifier.fit(valid_data)
        with pytest.raises(PredictionError):
            fitted_classifier.predict(invalid_data.features)
