import pytest
from safeds.data.tabular.containers import Table
from safeds.ml.classical.regression import RidgeRegression


def test_should_raise_if_alpha_is_negative() -> None:
    with pytest.raises(ValueError, match="alpha must be non-negative"):
        RidgeRegression(alpha=-1.0)


def test_should_warn_if_alpha_is_zero() -> None:
    with pytest.warns(
        UserWarning,
        match=(
            "Setting alpha to zero makes this model equivalent to LinearRegression. You "
            "should use LinearRegression instead for better numerical stability."
        ),
    ):
        RidgeRegression(alpha=0.0)


def test_should_pass_alpha_to_fitted_regressor() -> None:
    regressor = RidgeRegression(alpha=1.0)
    fitted_regressor = regressor.fit(Table.from_dict({"A": [1, 2, 4], "B": [1, 2, 3]}).tag_columns("B"))
    assert regressor._alpha == fitted_regressor._alpha


def test_should_pass_alpha_to_sklearn() -> None:
    regressor = RidgeRegression(alpha=1.0)
    fitted_regressor = regressor.fit(Table.from_dict({"A": [1, 2, 4], "B": [1, 2, 3]}).tag_columns("B"))
    assert fitted_regressor._wrapped_regressor is not None
    assert fitted_regressor._wrapped_regressor.alpha == fitted_regressor._alpha
