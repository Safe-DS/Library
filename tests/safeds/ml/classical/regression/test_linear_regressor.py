import sys

import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import OutOfBoundsError
from safeds.ml.classical.regression._linear_regressor import LinearRegressor, _Linear
from safeds.ml.hyperparameters import Choice


def penalties() -> list[LinearRegressor.Penalty]:
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

    @pytest.mark.parametrize(
        ("penalty1", "penalty2"),
        ([(x, y) for x in penalties() for y in penalties() if x.__class__ == y.__class__]),
        ids=lambda x: x.__class__.__name__,
    )
    def test_equal_penalties(
        self,
        penalty1: LinearRegressor.Penalty,
        penalty2: LinearRegressor.Penalty,
    ) -> None:
        assert penalty1 == penalty2

    @pytest.mark.parametrize(
        ("penalty1", "penalty2"),
        ([(x, y) for x in penalties() for y in penalties() if x.__class__ != y.__class__]),
        ids=lambda x: x.__class__.__name__,
    )
    def test_unequal_penalties(
        self,
        penalty1: LinearRegressor.Penalty,
        penalty2: LinearRegressor.Penalty,
    ) -> None:
        assert penalty1 != penalty2

    @pytest.mark.parametrize(
        ("penalty1", "penalty2"),
        ([(x, y) for x in penalties() for y in penalties() if x.__class__ == y.__class__]),
        ids=lambda x: x.__class__.__name__,
    )
    def test_should_return_same_hash_for_equal_penalties(
        self,
        penalty1: LinearRegressor.Penalty,
        penalty2: LinearRegressor.Penalty,
    ) -> None:
        assert hash(penalty1) == hash(penalty2)

    @pytest.mark.parametrize(
        ("penalty1", "penalty2"),
        ([(x, y) for x in penalties() for y in penalties() if x.__class__ != y.__class__]),
        ids=lambda x: x.__class__.__name__,
    )
    def test_should_return_different_hash_for_unequal_penalties(
        self,
        penalty1: LinearRegressor.Penalty,
        penalty2: LinearRegressor.Penalty,
    ) -> None:
        assert hash(penalty1) != hash(penalty2)

    @pytest.mark.parametrize(
        "penalty",
        ([LinearRegressor.Penalty.ridge(), LinearRegressor.Penalty.lasso(), LinearRegressor.Penalty.elastic_net()]),
        ids=lambda x: x.__class__.__name__,
    )
    def test_sizeof_kernel(
        self,
        penalty: LinearRegressor.Penalty,
    ) -> None:
        assert sys.getsizeof(penalty) > sys.getsizeof(object())

    class TestLinear:
        def test_str(self) -> None:
            linear_penalty = LinearRegressor.Penalty.linear()
            assert linear_penalty.__str__() == "Linear"

    class TestRidge:
        def test_str(self) -> None:
            ridge_penalty = LinearRegressor.Penalty.ridge(0.5)
            assert ridge_penalty.__str__() == f"Ridge(alpha={0.5})"

        @pytest.mark.parametrize("alpha", [-0.5, Choice(-0.5)], ids=["minus_zero_point_five", "invalid_choice"])
        def test_should_raise_if_alpha_out_of_bounds_ridge(self, alpha: float | Choice[float]) -> None:
            with pytest.raises(OutOfBoundsError):
                LinearRegressor(penalty=LinearRegressor.Penalty.ridge(alpha=alpha))

        def test_should_assert_alpha_is_set_correctly(self) -> None:
            alpha = 0.69
            assert LinearRegressor.Penalty.ridge(alpha=alpha).alpha == alpha

    class TestLasso:
        def test_str(self) -> None:
            lasso_penalty = LinearRegressor.Penalty.lasso(0.5)
            assert lasso_penalty.__str__() == f"Lasso(alpha={0.5})"

        @pytest.mark.parametrize("alpha", [-0.5, Choice(-0.5)], ids=["minus_zero_point_five", "invalid_choice"])
        def test_should_raise_if_alpha_out_of_bounds_lasso(self, alpha: float | Choice[float]) -> None:
            with pytest.raises(OutOfBoundsError):
                LinearRegressor(penalty=LinearRegressor.Penalty.lasso(alpha=alpha))

        def test_should_assert_alpha_is_set_correctly(self) -> None:
            alpha = 0.69
            assert LinearRegressor.Penalty.lasso(alpha=alpha).alpha == alpha

    class TestElasticNet:
        def test_str(self) -> None:
            elastic_net_penalty = LinearRegressor.Penalty.elastic_net(0.5, 0.75)
            assert elastic_net_penalty.__str__() == f"ElasticNet(alpha={0.5}, lasso_ratio={0.75})"

        @pytest.mark.parametrize("alpha", [-0.5, Choice(-0.5)], ids=["minus_zero_point_five", "invalid_choice"])
        def test_should_raise_if_alpha_out_of_bounds(self, alpha: float | Choice[float]) -> None:
            with pytest.raises(OutOfBoundsError):
                LinearRegressor(penalty=LinearRegressor.Penalty.elastic_net(alpha=alpha))

        @pytest.mark.parametrize(
            "lasso_ratio",
            [-0.5, 1.5, Choice(-0.5)],
            ids=["minus_zero_point_five", "one_point_five", "invalid_choice"],
        )
        def test_should_raise_if_lasso_ratio_out_of_bounds(self, lasso_ratio: float | Choice[float]) -> None:
            with pytest.raises(OutOfBoundsError):
                LinearRegressor(penalty=LinearRegressor.Penalty.elastic_net(lasso_ratio=lasso_ratio))

        def test_should_assert_alpha_is_set_correctly(self) -> None:
            alpha = 0.69
            lasso_ratio = 0.96
            elastic_pen = LinearRegressor.Penalty.elastic_net(alpha=alpha, lasso_ratio=lasso_ratio)
            assert elastic_pen.alpha == alpha
            assert elastic_pen.lasso_ratio == lasso_ratio

