import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import OutOfBoundsError
from safeds.ml.classical.regression import AdaBoostRegressor


@pytest.fixture()
def training_set() -> TaggedTable:
    table = Table({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    return table.tag_columns(target_name="col1", feature_names=["col2"])


class TestLearner:
    def test_should_be_passed_to_fitted_model(self, training_set: TaggedTable) -> None:
        learner = AdaBoostRegressor()
        fitted_model = AdaBoostRegressor(learner=learner).fit(training_set)
        assert fitted_model.learner == learner

    def test_should_be_passed_to_sklearn(self, training_set: TaggedTable) -> None:
        learner = AdaBoostRegressor()
        fitted_model = AdaBoostRegressor(learner=learner).fit(training_set)
        assert fitted_model._wrapped_regressor is not None
        assert isinstance(fitted_model._wrapped_regressor.estimator, type(learner._get_sklearn_regressor()))


class TestMaximumNumberOfLearners:
    def test_should_be_passed_to_fitted_model(self, training_set: TaggedTable) -> None:
        fitted_model = AdaBoostRegressor(maximum_number_of_learners=2).fit(training_set)
        assert fitted_model.maximum_number_of_learners == 2

    def test_should_be_passed_to_sklearn(self, training_set: TaggedTable) -> None:
        fitted_model = AdaBoostRegressor(maximum_number_of_learners=2).fit(training_set)
        assert fitted_model._wrapped_regressor is not None
        assert fitted_model._wrapped_regressor.n_estimators == 2

    @pytest.mark.parametrize("maximum_number_of_learners", [-1, 0], ids=["minus_one", "zero"])
    def test_should_raise_if_less_than_or_equal_to_0(self, maximum_number_of_learners: int) -> None:
        with pytest.raises(
            OutOfBoundsError,
            match=rf"maximum_number_of_learners \(={maximum_number_of_learners}\) is not inside \[1, \u221e\)\.",
        ):
            AdaBoostRegressor(maximum_number_of_learners=maximum_number_of_learners)


class TestLearningRate:
    def test_should_be_passed_to_fitted_model(self, training_set: TaggedTable) -> None:
        fitted_model = AdaBoostRegressor(learning_rate=2).fit(training_set)
        assert fitted_model.learning_rate == 2

    def test_should_be_passed_to_sklearn(self, training_set: TaggedTable) -> None:
        fitted_model = AdaBoostRegressor(learning_rate=2).fit(training_set)
        assert fitted_model._wrapped_regressor is not None
        assert fitted_model._wrapped_regressor.learning_rate == 2

    @pytest.mark.parametrize("learning_rate", [-1.0, 0.0], ids=["minus_one", "zero"])
    def test_should_raise_if_less_than_or_equal_to_0(self, learning_rate: float) -> None:
        with pytest.raises(
            OutOfBoundsError,
            match=rf"learning_rate \(={learning_rate}\) is not inside \(0, \u221e\)\.",
        ):
            AdaBoostRegressor(learning_rate=learning_rate)
