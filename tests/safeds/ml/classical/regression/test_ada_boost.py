import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.ml.classical.regression import AdaBoost


@pytest.fixture()
def training_set() -> TaggedTable:
    table = Table({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    return table.tag_columns(target_name="col1", feature_names=["col2"])


class TestLearner:
    def test_should_be_passed_to_fitted_model(self, training_set: TaggedTable) -> None:
        learner = AdaBoost()
        fitted_model = AdaBoost(learner=learner).fit(training_set)
        assert fitted_model.learner == learner

    def test_should_be_passed_to_sklearn(self, training_set: TaggedTable) -> None:
        learner = AdaBoost()
        fitted_model = AdaBoost(learner=learner).fit(training_set)
        assert fitted_model._wrapped_regressor is not None
        assert isinstance(fitted_model._wrapped_regressor.estimator, type(learner._get_sklearn_regressor()))


class TestMaximumNumberOfLearners:
    def test_should_be_passed_to_fitted_model(self, training_set: TaggedTable) -> None:
        fitted_model = AdaBoost(maximum_number_of_learners=2).fit(training_set)
        assert fitted_model.maximum_number_of_learners == 2

    def test_should_be_passed_to_sklearn(self, training_set: TaggedTable) -> None:
        fitted_model = AdaBoost(maximum_number_of_learners=2).fit(training_set)
        assert fitted_model._wrapped_regressor is not None
        assert fitted_model._wrapped_regressor.n_estimators == 2

    def test_should_raise_if_less_than_or_equal_to_0(self) -> None:
        with pytest.raises(ValueError, match="The parameter 'maximum_number_of_learners' has to be grater than 0."):
            AdaBoost(maximum_number_of_learners=-1)


class TestLearningRate:
    def test_should_be_passed_to_fitted_model(self, training_set: TaggedTable) -> None:
        fitted_model = AdaBoost(learning_rate=2).fit(training_set)
        assert fitted_model.learning_rate == 2

    def test_should_be_passed_to_sklearn(self, training_set: TaggedTable) -> None:
        fitted_model = AdaBoost(learning_rate=2).fit(training_set)
        assert fitted_model._wrapped_regressor is not None
        assert fitted_model._wrapped_regressor.learning_rate == 2

    def test_should_raise_if_less_than_or_equal_to_0(self) -> None:
        with pytest.raises(ValueError, match="The parameter 'learning_rate' has to be greater than 0."):
            AdaBoost(learning_rate=-1)
