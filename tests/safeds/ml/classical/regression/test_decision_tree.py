import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import ModelNotFittedError, OutOfBoundsError
from safeds.ml.classical.regression import DecisionTreeRegressor
from syrupy import SnapshotAssertion


@pytest.fixture()
def training_set() -> TabularDataset:
    table = Table({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    return table.to_tabular_dataset(target_name="col1")


class TestMaxDepth:
    def test_should_be_passed_to_fitted_model(self, training_set: TabularDataset) -> None:
        fitted_model = DecisionTreeRegressor(max_depth=2).fit(training_set)
        assert fitted_model.max_depth == 2

    def test_should_be_passed_to_sklearn(self, training_set: TabularDataset) -> None:
        fitted_model = DecisionTreeRegressor(max_depth=2).fit(training_set)
        assert fitted_model._wrapped_model is not None
        assert fitted_model._wrapped_model.max_depth == 2

    @pytest.mark.parametrize("max_depth", [-1, 0], ids=["minus_one", "zero"])
    def test_should_raise_if_less_than_or_equal_to_0(self, max_depth: int) -> None:
        with pytest.raises(OutOfBoundsError):
            DecisionTreeRegressor(max_depth=max_depth)


class TestMinSampleCountInLeaves:
    def test_should_be_passed_to_fitted_model(self, training_set: TabularDataset) -> None:
        fitted_model = DecisionTreeRegressor(min_sample_count_in_leaves=2).fit(training_set)
        assert fitted_model.min_sample_count_in_leaves == 2

    def test_should_be_passed_to_sklearn(self, training_set: TabularDataset) -> None:
        fitted_model = DecisionTreeRegressor(min_sample_count_in_leaves=2).fit(training_set)
        assert fitted_model._wrapped_model is not None
        assert fitted_model._wrapped_model.min_samples_leaf == 2

    @pytest.mark.parametrize("min_sample_count_in_leaves", [-1, 0], ids=["minus_one", "zero"])
    def test_should_raise_if_less_than_or_equal_to_0(self, min_sample_count_in_leaves: int) -> None:
        with pytest.raises(OutOfBoundsError):
            DecisionTreeRegressor(min_sample_count_in_leaves=min_sample_count_in_leaves)


class TestPlot:
    def test_should_raise_if_model_is_not_fittet(self) -> None:
        model = DecisionTreeRegressor()
        with pytest.raises(ModelNotFittedError):
            model.plot()

    def test_should_check_that_plot_image_is_same_as_plt_figure(
        self,
        training_set: TabularDataset,
        snapshot_png_image: SnapshotAssertion,
    ) -> None:
        fitted_model = DecisionTreeRegressor().fit(training_set)
        image = fitted_model.plot()
        assert image == snapshot_png_image
