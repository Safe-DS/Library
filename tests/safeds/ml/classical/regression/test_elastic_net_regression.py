import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.ml.classical.regression import ElasticNetRegression


@pytest.fixture()
def training_set() -> TaggedTable:
    table = Table.from_dict({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    return table.tag_columns(target_name="col1", feature_names=["col2"])


class TestAlpha:
    def test_should_be_passed_to_fitted_model(self, training_set: TaggedTable) -> None:
        fitted_model = ElasticNetRegression(alpha=1).fit(training_set)
        assert fitted_model._alpha == 1

    def test_should_be_passed_to_sklearn(self, training_set: TaggedTable) -> None:
        fitted_model = ElasticNetRegression(alpha=1).fit(training_set)
        assert fitted_model._wrapped_regressor is not None
        assert fitted_model._wrapped_regressor.alpha == 1

    def test_should_raise_if_less_than_0(self) -> None:
        with pytest.raises(ValueError, match="alpha must be non-negative"):
            ElasticNetRegression(alpha=-1)

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
        assert fitted_model._lasso_ratio == 0.3

    def test_should_be_passed_to_sklearn(self, training_set: TaggedTable) -> None:
        fitted_model = ElasticNetRegression(lasso_ratio=0.3).fit(training_set)
        assert fitted_model._wrapped_regressor is not None
        assert fitted_model._wrapped_regressor.l1_ratio == 0.3

    def test_should_raise_if_less_than_0(self) -> None:
        with pytest.raises(ValueError, match="lasso_ratio must be between 0 and 1."):
            ElasticNetRegression(lasso_ratio=-1.0)

    def test_should_raise_if_greater_than_1(self) -> None:
        with pytest.raises(ValueError, match="lasso_ratio must be between 0 and 1."):
            ElasticNetRegression(lasso_ratio=2.0)

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
