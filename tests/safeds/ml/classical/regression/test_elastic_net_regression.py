import pytest
from safeds.data.tabular.containers import Table
from safeds.ml.classical.regression import ElasticNetRegression


def test_alpha_invalid() -> None:
    with pytest.raises(ValueError, match="alpha must be positive"):
        ElasticNetRegression(alpha=-1.0)


def test_alpha_warning() -> None:
    with pytest.warns(UserWarning, match="Setting alpha to zero makes this model equivalent to LinearRegression. You "
                                         "should use LinearRegression instead for better numerical stability."):
        ElasticNetRegression(alpha=0.0)


def test_alpha_valid() -> None:
    training_set = Table.from_dict({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    tagged_training_set = training_set.tag_columns(target_name="col1", feature_names=["col2"])

    elastic_net_regression = ElasticNetRegression(alpha=1.0).fit(tagged_training_set)
    assert elastic_net_regression._wrapped_regressor is not None
    assert elastic_net_regression._wrapped_regressor.alpha == elastic_net_regression._alpha


def test_lasso_ratio_valid() -> None:
    training_set = Table.from_dict({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    tagged_training_set = training_set.tag_columns(target_name="col1", feature_names=["col2"])
    lasso_ratio = 0.3

    elastic_net_regression = ElasticNetRegression(lasso_ratio=lasso_ratio).fit(tagged_training_set)
    assert elastic_net_regression._wrapped_regressor is not None
    assert elastic_net_regression._wrapped_regressor.l1_ratio == lasso_ratio


def test_lasso_ratio_invalid() -> None:
    with pytest.raises(ValueError, match="lasso_ratio must be between 0 and 1."):
        ElasticNetRegression(lasso_ratio=-1.0)


def test_lasso_ratio_zero() -> None:
    with pytest.warns(
        UserWarning,
        match="ElasticNetRegression with lasso_ratio = 0 is essentially RidgeRegression."
        " Use RidgeRegression instead for better numerical stability.",
    ):
        ElasticNetRegression(lasso_ratio=0)


def test_lasso_ratio_one() -> None:
    with pytest.warns(
        UserWarning,
        match="ElasticNetRegression with lasso_ratio = 0 is essentially LassoRegression."
        " Use LassoRegression instead for better numerical stability.",
    ):
        ElasticNetRegression(lasso_ratio=1)


# (Default parameter is tested in `test_regressor.py`.)
