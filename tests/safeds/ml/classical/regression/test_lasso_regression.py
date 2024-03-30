import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import OutOfBoundsError
from safeds.ml.classical.regression import LassoRegressor


@pytest.fixture()
def training_set() -> TaggedTable:
    table = Table({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    return table.tag_columns(target_name="col1", feature_names=["col2"])


class TestAlpha:
    def test_should_be_passed_to_fitted_model(self, training_set: TaggedTable) -> None:
        fitted_model = LassoRegressor(alpha=1).fit(training_set)
        assert fitted_model.alpha == 1

    def test_should_be_passed_to_sklearn(self, training_set: TaggedTable) -> None:
        fitted_model = LassoRegressor(alpha=1).fit(training_set)
        assert fitted_model._wrapped_regressor is not None
        assert fitted_model._wrapped_regressor.alpha == 1

    @pytest.mark.parametrize("alpha", [-0.5], ids=["minus_zero_point_5"])
    def test_should_raise_if_less_than_0(self, alpha: float) -> None:
        with pytest.raises(OutOfBoundsError, match=rf"alpha \(={alpha}\) is not inside \[0, \u221e\)\."):
            LassoRegressor(alpha=alpha)

    def test_should_warn_if_equal_to_0(self) -> None:
        with pytest.warns(
            UserWarning,
            match=(
                "Setting alpha to zero makes this model equivalent to LinearRegression. You "
                "should use LinearRegression instead for better numerical stability."
            ),
        ):
            LassoRegressor(alpha=0)
