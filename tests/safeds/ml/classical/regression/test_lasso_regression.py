import pytest
from safeds.data.tabular.containers import Table
from safeds.ml.classical.regression import LassoRegression


def test_should_throw_value_error() -> None:
    with pytest.raises(ValueError, match="alpha must be non-negative"):
        LassoRegression(alpha=-1)


def test_should_throw_warning() -> None:
    with pytest.warns(
        UserWarning,
        match=(
            "Setting alpha to zero makes this model equivalent to LinearRegression. You "
            "should use LinearRegression instead for better numerical stability."
        ),
    ):
        LassoRegression(alpha=0)


def test_should_give_alpha_to_sklearn() -> None:
    training_set = Table.from_dict({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    tagged_table = training_set.tag_columns("col1")

    regressor = LassoRegression(alpha=1).fit(tagged_table)
    assert regressor._wrapped_regressor is not None
    assert regressor._wrapped_regressor.alpha == regressor._alpha
