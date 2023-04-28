import pytest
from safeds.data.tabular.containers import Table
from safeds.ml.classical.regression import RidgeRegression


def test_ridge_regression_invalid() -> None:
    with pytest.raises(ValueError, match="The alpha parameter should not be below 0"):
        RidgeRegression(alpha=-1.0)


def test_ridge_regression_warning() -> None:
    with pytest.warns(
        UserWarning,
        match=(
            "RidgeRegression with alpha 0 performs the exact way as LinearRegression. Please use LinearRegression"
            " instead."
        ),
    ):
        RidgeRegression(alpha=0.0)


def test_ridge_regression() -> None:
    regression = RidgeRegression(alpha=1.0)
    fitted_regression = regression.fit(Table.from_dict({"A": [1, 2, 4], "B": [1, 2, 3]}).tag_columns("B"))
    assert regression.alpha == fitted_regression.alpha
