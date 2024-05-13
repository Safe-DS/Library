import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import OutOfBoundsError
from safeds.ml.classical.regression import AdaBoostRegressor


@pytest.fixture()
def training_set() -> TabularDataset:
    table = Table({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    return table.to_tabular_dataset(target_name="col1")


class TestLearner:
    def test_should_be_passed_to_fitted_model(self, training_set: TabularDataset) -> None:
        learner = AdaBoostRegressor()
        fitted_model = AdaBoostRegressor(learner=learner).fit(training_set)
        assert fitted_model.learner == learner

    def test_should_be_passed_to_sklearn(self, training_set: TabularDataset) -> None:
        learner = AdaBoostRegressor()
        fitted_model = AdaBoostRegressor(learner=learner).fit(training_set)
        assert fitted_model._wrapped_model is not None
        assert isinstance(fitted_model._wrapped_model.estimator, type(learner._get_sklearn_model()))


class TestMaximumNumberOfLearners:
    def test_should_be_passed_to_fitted_model(self, training_set: TabularDataset) -> None:
        fitted_model = AdaBoostRegressor(maximum_number_of_learners=2).fit(training_set)
        assert fitted_model.maximum_number_of_learners == 2

    def test_should_be_passed_to_sklearn(self, training_set: TabularDataset) -> None:
        fitted_model = AdaBoostRegressor(maximum_number_of_learners=2).fit(training_set)
        assert fitted_model._wrapped_model is not None
        assert fitted_model._wrapped_model.n_estimators == 2

    @pytest.mark.parametrize("maximum_number_of_learners", [-1, 0], ids=["minus_one", "zero"])
    def test_should_raise_if_less_than_or_equal_to_0(self, maximum_number_of_learners: int) -> None:
        with pytest.raises(OutOfBoundsError):
            AdaBoostRegressor(maximum_number_of_learners=maximum_number_of_learners)


class TestLearningRate:
    def test_should_be_passed_to_fitted_model(self, training_set: TabularDataset) -> None:
        fitted_model = AdaBoostRegressor(learning_rate=2).fit(training_set)
        assert fitted_model.learning_rate == 2

    def test_should_be_passed_to_sklearn(self, training_set: TabularDataset) -> None:
        fitted_model = AdaBoostRegressor(learning_rate=2).fit(training_set)
        assert fitted_model._wrapped_model is not None
        assert fitted_model._wrapped_model.learning_rate == 2

    @pytest.mark.parametrize("learning_rate", [-1.0, 0.0], ids=["minus_one", "zero"])
    def test_should_raise_if_less_than_or_equal_to_0(self, learning_rate: float) -> None:
        with pytest.raises(OutOfBoundsError):
            AdaBoostRegressor(learning_rate=learning_rate)
