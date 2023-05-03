import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.ml.classical.regression import ElasticNetRegression


def test_should_give_lasso_ratio_to_sklearn() -> None:
    training_set = Table.from_dict({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    tagged_training_set = training_set.tag_columns(target_name="col1", feature_names=["col2"])
    lasso_ratio = 0.3

    elastic_net_regression = ElasticNetRegression(lasso_ratio=lasso_ratio).fit(tagged_training_set)
    assert elastic_net_regression._wrapped_regressor is not None
    assert elastic_net_regression._wrapped_regressor.l1_ratio == lasso_ratio


def test_should_throw_value_error_lasso_ratio() -> None:
    with pytest.raises(ValueError, match="lasso_ratio must be between 0 and 1."):
        ElasticNetRegression(lasso_ratio=-1.0)


def test_should_throw_warning_lasso_ratio_zero() -> None:
    with pytest.warns(
        UserWarning,
        match=(
            "ElasticNetRegression with lasso_ratio = 0 is essentially RidgeRegression."
            " Use RidgeRegression instead for better numerical stability."
        ),
    ):
        ElasticNetRegression(lasso_ratio=0)


def test_should_throw_warning_lasso_ratio_one() -> None:
    with pytest.warns(
        UserWarning,
        match=(
            "ElasticNetRegression with lasso_ratio = 0 is essentially LassoRegression."
            " Use LassoRegression instead for better numerical stability."
        ),
    ):
        ElasticNetRegression(lasso_ratio=1)


# --------------------------------------------------------------------------------------------

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
