import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import OutOfBoundsError
from safeds.ml.classical.regression import ElasticNetRegression


@pytest.fixture()
def training_set() -> TaggedTable:
    table = Table({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    return table.tag_columns(target_name="col1", feature_names=["col2"])


class TestAlpha:
    def test_should_be_passed_to_fitted_model(self, training_set: TaggedTable) -> None:
        fitted_model = ElasticNetRegression(alpha=1).fit(training_set)
        assert fitted_model.alpha == 1

    def test_should_be_passed_to_sklearn(self, training_set: TaggedTable) -> None:
        fitted_model = ElasticNetRegression(alpha=1).fit(training_set)
        assert fitted_model._wrapped_regressor is not None
        assert fitted_model._wrapped_regressor.alpha == 1

    @pytest.mark.parametrize("alpha", [-0.5], ids=["minus_0_point_5"])
    def test_should_raise_if_less_than_0(self, alpha: float) -> None:
        with pytest.raises(
            OutOfBoundsError,
            match=rf"alpha \(={alpha}\) is not inside \[0, \u221e\)\.",
        ):
            ElasticNetRegression(alpha=alpha)

    def test_should_warn_if_equal_to_0(self) -> None:
        with pytest.warns(
            UserWarning,
            match=(
                "Setting alpha to zero makes this model equivalent to LinearRegression. You "
                "should use LinearRegression instead for better numerical stability."
            ),
        ):
            ElasticNetRegression(alpha=0)


class TestLassoRatio:
    def test_should_be_passed_to_fitted_model(self, training_set: TaggedTable) -> None:
        fitted_model = ElasticNetRegression(lasso_ratio=0.3).fit(training_set)
        assert fitted_model.lasso_ratio == 0.3

    def test_should_be_passed_to_sklearn(self, training_set: TaggedTable) -> None:
        fitted_model = ElasticNetRegression(lasso_ratio=0.3).fit(training_set)
        assert fitted_model._wrapped_regressor is not None
        assert fitted_model._wrapped_regressor.l1_ratio == 0.3

    @pytest.mark.parametrize("lasso_ratio", [-0.5, 1.5], ids=["minus_zero_point_5", "one_point_5"])
    def test_should_raise_if_not_between_0_and_1(self, lasso_ratio: float) -> None:
        with pytest.raises(
            OutOfBoundsError,
            match=rf"lasso_ratio \(={lasso_ratio}\) is not inside \[0, 1\]\.",
        ):
            ElasticNetRegression(lasso_ratio=lasso_ratio)

    def test_should_warn_if_0(self) -> None:
        with pytest.warns(
            UserWarning,
            match=(
                "ElasticNetRegression with lasso_ratio = 0 is essentially RidgeRegression."
                " Use RidgeRegression instead for better numerical stability."
            ),
        ):
            ElasticNetRegression(lasso_ratio=0)

    def test_should_warn_if_1(self) -> None:
        with pytest.warns(
            UserWarning,
            match=(
                "ElasticNetRegression with lasso_ratio = 0 is essentially LassoRegression."
                " Use LassoRegression instead for better numerical stability."
            ),
        ):
            ElasticNetRegression(lasso_ratio=1)
