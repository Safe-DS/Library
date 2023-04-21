import warnings

import pytest

from safeds.data.tabular.containers import Table
from safeds.ml.classical.regression import RidgeRegression


def test_ridge_regression_invalid() -> None:
    with pytest.raises(ValueError):
        RidgeRegression(alpha=-1.0)


def test_ridge_regression_warning() -> None:
    with pytest.warns():
        RidgeRegression(alpha=0.0)


def test_ridge_regression() -> None:
    regression = RidgeRegression(alpha=1.0)
    fitted_regression = regression.fit(
        Table.from_dict({"A": [1, 2, 4], "B": [1, 2, 3]}).tag_columns("B"))
    assert regression.alpha == fitted_regression.alpha


def test_warning() -> None:
    with pytest.warns(UserWarning):
        warnings.warn(UserWarning("Test"))
