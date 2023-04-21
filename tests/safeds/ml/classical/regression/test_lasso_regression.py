import pytest

from safeds.data.tabular.containers import Table
from safeds.ml.classical.regression import LassoRegression


def test_alpha_invalid() -> None:
    with pytest.raises(ValueError, match="alpha must be non-negative"):
        LassoRegression(alpha=-1)


def test_alpha_warning() -> None:
    with pytest.warns(UserWarning, match="alpha is zero, you should use LinearRegression instead"):
        LassoRegression(alpha=0)


def test_alpha_valid() -> None:
    training_set = Table.from_dict({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    tagged_table = training_set.tag_columns("col1")

    regressor = LassoRegression(alpha=1).fit(tagged_table)
    assert regressor._wrapped_regressor is not None
    assert regressor._wrapped_regressor.alpha == regressor._alpha
