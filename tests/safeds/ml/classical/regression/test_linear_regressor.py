import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import OutOfBoundsError
from safeds.ml.classical.regression._linear_regressor import LinearRegressor, _Linear
from safeds.ml.hyperparameters import Choice


def kernels() -> list[LinearRegressor.Penalty]:
    """
    Return the list of penalties to test.

    After you implemented a new penalty, add it to this list to ensure its `__hash__` and `__eq__` method work as
    expected.

    Returns
    -------
    penalties:
        The list of penalties to test.
    """
    return [
        LinearRegressor.Penalty.linear(),
        LinearRegressor.Penalty.ridge(),
        LinearRegressor.Penalty.lasso(),
        LinearRegressor.Penalty.elastic_net(),
    ]


@pytest.fixture()
def training_set() -> TabularDataset:
    table = Table({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    return table.to_tabular_dataset(target_name="col1")


class TestPenalty:
    def test_should_be_passed_to_fitted_model(self, training_set: TabularDataset) -> None:
        penalty = LinearRegressor.Penalty.linear()
        fitted_model = LinearRegressor(penalty=penalty).fit(training_set=training_set)
        assert isinstance(fitted_model.penalty, _Linear)
        assert fitted_model._wrapped_model is not None

    @pytest.mark.parametrize("alpha", [-0.5, Choice(-0.5)], ids=["minus_0_point_5", "invalid_choice"])
    def test_should_raise_if_alpha_out_of_bounds_ridge(self, alpha: float | Choice[float]) -> None:
        with pytest.raises(OutOfBoundsError):
            LinearRegressor(penalty=LinearRegressor.Penalty.ridge(alpha=alpha))

    @pytest.mark.parametrize("alpha", [-0.5, Choice(-0.5)], ids=["minus_0_point_5", "invalid_choice"])
    def test_should_raise_if_alpha_out_of_bounds_lasso(self, alpha: float | Choice[float]) -> None:
        with pytest.raises(OutOfBoundsError):
            LinearRegressor(penalty=LinearRegressor.Penalty.lasso(alpha=alpha))

    @pytest.mark.parametrize("alpha", [-0.5, Choice(-0.5)], ids=["minus_0_point_5", "invalid_choice"])
    def test_should_raise_if_alpha_out_of_bounds_elastic_net(self, alpha: float | Choice[float]) -> None:
        with pytest.raises(OutOfBoundsError):
            LinearRegressor(penalty=LinearRegressor.Penalty.elastic_net(alpha=alpha))

    @pytest.mark.parametrize(
        "lasso_ratio",
        [-0.5, 1.5, Choice(-0.5)],
        ids=["minus_0_point_5", "one_point_five", "invalid_choice"],
    )
    def test_should_raise_if_lasso_ratio_out_of_bounds_elastic_net(self, lasso_ratio: float | Choice[float]) -> None:
        with pytest.raises(OutOfBoundsError):
            LinearRegressor(penalty=LinearRegressor.Penalty.elastic_net(lasso_ratio=lasso_ratio))
