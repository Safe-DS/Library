import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import OutOfBoundsError
from safeds.ml.classical.regression import GradientBoostingRegressor
from safeds.ml.hyperparameters import Choice


@pytest.fixture()
def training_set() -> TabularDataset:
    table = Table({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    return table.to_tabular_dataset(target_name="col1")


class TestTreeCount:
    def test_should_be_passed_to_fitted_model(self, training_set: TabularDataset) -> None:
        fitted_model = GradientBoostingRegressor(tree_count=2).fit(training_set)
        assert fitted_model.tree_count == 2

    def test_should_be_passed_to_sklearn(self, training_set: TabularDataset) -> None:
        fitted_model = GradientBoostingRegressor(tree_count=2).fit(training_set)
        assert fitted_model._wrapped_model is not None
        assert fitted_model._wrapped_model.n_estimators == 2

    @pytest.mark.parametrize("tree_count", [-1, 0, Choice(-1)], ids=["minus_one", "zero", "invalid_choice"])
    def test_should_raise_if_less_than_1(self, tree_count: int | Choice[int]) -> None:
        with pytest.raises(OutOfBoundsError):
            GradientBoostingRegressor(tree_count=tree_count)


class TestLearningRate:
    def test_should_be_passed_to_fitted_model(self, training_set: TabularDataset) -> None:
        fitted_model = GradientBoostingRegressor(learning_rate=2).fit(training_set)
        assert fitted_model.learning_rate == 2

    def test_should_be_passed_to_sklearn(self, training_set: TabularDataset) -> None:
        fitted_model = GradientBoostingRegressor(learning_rate=2).fit(training_set)
        assert fitted_model._wrapped_model is not None
        assert fitted_model._wrapped_model.learning_rate == 2

    @pytest.mark.parametrize("learning_rate", [-1.0, 0.0, Choice(-1.0)], ids=["minus_one", "zero", "invalid_choice"])
    def test_should_raise_if_less_than_or_equal_to_0(self, learning_rate: float | Choice[float]) -> None:
        with pytest.raises(OutOfBoundsError):
            GradientBoostingRegressor(learning_rate=learning_rate)
