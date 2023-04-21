import warnings

import pytest
from safeds.data.tabular.containers import Table
from safeds.ml.classical.regression._elastic_net_regression import ElasticNetRegression


# def test_lasso_ratio_default() -> None: is in test_regressor.py


def test_lasso_ratio_valid() -> None:
    training_set = Table.from_dict({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    tagged_training_set = training_set.tag_columns(target_name="col1", feature_names=["col2"])
    lasso_ratio = .3

    elastic_net_regression = ElasticNetRegression(lasso_ratio).fit(tagged_training_set)
    assert elastic_net_regression._wrapped_regressor is not None
    assert elastic_net_regression._wrapped_regressor.l1_ratio == lasso_ratio


def test_lasso_ratio_invalid() -> None:
    with pytest.raises(ValueError, match="lasso_ratio must be between 0 and 1."):
        ElasticNetRegression(-1)


def test_lasso_ratio_zero() -> None:
    with pytest.warns(UserWarning, match="ElasticNetRegression with lasso_ratio = 0 is essentially RidgeRegression."
                                         " Use RidgeRegression instead for better numerical stability."):
        ElasticNetRegression(0)


def test_lasso_ratio_one() -> None:
    with pytest.warns(UserWarning, match="ElasticNetRegression with lasso_ratio = 0 is essentially LassoRegression."
                                         " Use LassoRegression instead for better numerical stability."):
        ElasticNetRegression(1)
