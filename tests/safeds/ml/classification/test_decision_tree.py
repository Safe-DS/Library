import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import LearningError, PredictionError
from safeds.ml.classification import Classifier, DecisionTree
from tests.helpers import resolve_resource_path


@pytest.fixture()
def classifier() -> Classifier:
    return DecisionTree()


@pytest.fixture()
def valid_data() -> TaggedTable:
    table = Table.from_csv_file(resolve_resource_path("test_decision_tree.csv"))
    return table.tag_columns(target_name="T")


@pytest.fixture()
def invalid_data() -> TaggedTable:
    table = Table.from_csv_file(resolve_resource_path("test_decision_tree_invalid.csv"))
    return table.tag_columns(target_name="T")


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

    def test_should_set_correct_target_name(self, classifier: Classifier, valid_data: TaggedTable) -> None:
        fitted_classifier = classifier.fit(valid_data)
        prediction = fitted_classifier.predict(valid_data.features)
        assert prediction.target.name == "T"

    def test_should_raise_when_not_fitted(self, classifier: Classifier, valid_data: TaggedTable) -> None:
        with pytest.raises(PredictionError):
            classifier.predict(valid_data.features)

    def test_should_raise_on_invalid_data(
        self, classifier: Classifier, valid_data: TaggedTable, invalid_data: TaggedTable
    ) -> None:
        fitted_classifier = classifier.fit(valid_data)
        with pytest.raises(PredictionError):
            fitted_classifier.predict(invalid_data.features)
