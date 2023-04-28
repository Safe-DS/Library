import pytest
from safeds.data.tabular.containers import Table
from safeds.ml.classical.regression import ElasticNetRegression


def test_should_throw_value_error_alpha() -> None:
    with pytest.raises(ValueError, match="alpha must be non-negative"):
        ElasticNetRegression(alpha=-1.0)


def test_should_throw_warning_alpha() -> None:
    with pytest.warns(
        UserWarning,
        match=(
            "Setting alpha to zero makes this model equivalent to LinearRegression. You "
            "should use LinearRegression instead for better numerical stability."
        ),
    ):
        ElasticNetRegression(alpha=0.0)


def test_should_give_alpha_to_sklearn() -> None:
    training_set = Table.from_dict({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    tagged_training_set = training_set.tag_columns(target_name="col1", feature_names=["col2"])

    elastic_net_regression = ElasticNetRegression(alpha=1.0).fit(tagged_training_set)
    assert elastic_net_regression._wrapped_regressor is not None
    assert elastic_net_regression._wrapped_regressor.alpha == elastic_net_regression._alpha


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


# (Default parameter is tested in `test_regressor.py`.)
