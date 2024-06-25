import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import OutOfBoundsError
from safeds.ml.classical.regression import ElasticNetRegressor
from safeds.ml.hyperparameters import Choice


@pytest.fixture()
def training_set() -> TabularDataset:
    table = Table({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    return table.to_tabular_dataset(target_name="col1")


class TestAlpha:
    def test_should_be_passed_to_fitted_model(self, training_set: TabularDataset) -> None:
        fitted_model = ElasticNetRegressor(alpha=1).fit(training_set)
        assert fitted_model.alpha == 1

    def test_should_be_passed_to_sklearn(self, training_set: TabularDataset) -> None:
        fitted_model = ElasticNetRegressor(alpha=1).fit(training_set)
        assert fitted_model._wrapped_model is not None
        assert fitted_model._wrapped_model.alpha == 1

    @pytest.mark.parametrize("alpha", [-0.5, Choice(-0.5)], ids=["minus_0_point_5", "invalid_choice"])
    def test_should_raise_if_less_than_0(self, alpha: float | Choice[float]) -> None:
        with pytest.raises(OutOfBoundsError):
            ElasticNetRegressor(alpha=alpha)


class TestLassoRatio:
    def test_should_be_passed_to_fitted_model(self, training_set: TabularDataset) -> None:
        fitted_model = ElasticNetRegressor(lasso_ratio=0.3).fit(training_set)
        assert fitted_model.lasso_ratio == 0.3

    def test_should_be_passed_to_sklearn(self, training_set: TabularDataset) -> None:
        fitted_model = ElasticNetRegressor(lasso_ratio=0.3).fit(training_set)
        assert fitted_model._wrapped_model is not None
        assert fitted_model._wrapped_model.l1_ratio == 0.3

    @pytest.mark.parametrize(
        "lasso_ratio", [-0.5, 1.5, Choice(-0.5)], ids=["minus_zero_point_5", "one_point_5", "invalid_choice"],
    )
    def test_should_raise_if_not_between_0_and_1(self, lasso_ratio: float | Choice[float]) -> None:
        with pytest.raises(OutOfBoundsError):
            ElasticNetRegressor(lasso_ratio=lasso_ratio)
