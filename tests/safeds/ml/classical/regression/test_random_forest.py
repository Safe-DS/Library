import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import OutOfBoundsError
from safeds.ml.classical.regression import RandomForestRegressor
from safeds.ml.hyperparameters import Choice


@pytest.fixture()
def training_set() -> TabularDataset:
    table = Table({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    return table.to_tabular_dataset("col1")


class TestTreeCount:
    def test_should_be_passed_to_fitted_model(self, training_set: TabularDataset) -> None:
        fitted_model = RandomForestRegressor(tree_count=2).fit(training_set)
        assert fitted_model.tree_count == 2

    def test_should_be_passed_to_sklearn(self, training_set: TabularDataset) -> None:
        fitted_model = RandomForestRegressor(tree_count=2).fit(training_set)
        assert fitted_model._wrapped_model is not None
        assert fitted_model._wrapped_model.n_estimators == 2

    @pytest.mark.parametrize("tree_count", [-1, 0, Choice(-1)], ids=["minus_one", "zero", "invalid_choice"])
    def test_should_raise_if_less_than_or_equal_to_0(self, tree_count: int | Choice[int]) -> None:
        with pytest.raises(OutOfBoundsError):
            RandomForestRegressor(tree_count=tree_count)


class TestMaxDepth:
    def test_should_be_passed_to_fitted_model(self, training_set: TabularDataset) -> None:
        fitted_model = RandomForestRegressor(max_depth=2).fit(training_set)
        assert fitted_model.max_depth == 2

    def test_should_be_passed_to_sklearn(self, training_set: TabularDataset) -> None:
        fitted_model = RandomForestRegressor(max_depth=2).fit(training_set)
        assert fitted_model._wrapped_model is not None
        assert fitted_model._wrapped_model.max_depth == 2

    @pytest.mark.parametrize("max_depth", [-1, 0, Choice(-1)], ids=["minus_one", "zero", "invalid_choice"])
    def test_should_raise_if_less_than_or_equal_to_0(self, max_depth: int | None | Choice[int | None]) -> None:
        with pytest.raises(OutOfBoundsError):
            RandomForestRegressor(max_depth=max_depth)


class TestMinSampleCountInLeaves:
    def test_should_be_passed_to_fitted_model(self, training_set: TabularDataset) -> None:
        fitted_model = RandomForestRegressor(min_sample_count_in_leaves=2).fit(training_set)
        assert fitted_model.min_sample_count_in_leaves == 2

    def test_should_be_passed_to_sklearn(self, training_set: TabularDataset) -> None:
        fitted_model = RandomForestRegressor(min_sample_count_in_leaves=2).fit(training_set)
        assert fitted_model._wrapped_model is not None
        assert fitted_model._wrapped_model.min_samples_leaf == 2

    @pytest.mark.parametrize(
        "min_sample_count_in_leaves",
        [-1, 0, Choice(-1)],
        ids=["minus_one", "zero", "invalid_choice"],
    )
    def test_should_raise_if_less_than_or_equal_to_0(self, min_sample_count_in_leaves: int | Choice[int]) -> None:
        with pytest.raises(OutOfBoundsError):
            RandomForestRegressor(min_sample_count_in_leaves=min_sample_count_in_leaves)
