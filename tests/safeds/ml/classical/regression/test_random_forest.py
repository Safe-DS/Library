import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import OutOfBoundsError
from safeds.ml.classical.regression import RandomForestRegressor


@pytest.fixture()
def training_set() -> TabularDataset:
    table = Table({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    return table.to_tabular_dataset(target_name="col1")


class TestNumberOfTrees:
    def test_should_be_passed_to_fitted_model(self, training_set: TabularDataset) -> None:
        fitted_model = RandomForestRegressor(number_of_trees=2).fit(training_set)
        assert fitted_model.number_of_trees == 2

    def test_should_be_passed_to_sklearn(self, training_set: TabularDataset) -> None:
        fitted_model = RandomForestRegressor(number_of_trees=2).fit(training_set)
        assert fitted_model._wrapped_regressor is not None
        assert fitted_model._wrapped_regressor.n_estimators == 2

    @pytest.mark.parametrize("number_of_trees", [-1, 0], ids=["minus_one", "zero"])
    def test_should_raise_if_less_than_or_equal_to_0(self, number_of_trees: int) -> None:
        with pytest.raises(
            OutOfBoundsError,
            match=rf"number_of_trees \(={number_of_trees}\) is not inside \[1, \u221e\)\.",
        ):
            RandomForestRegressor(number_of_trees=number_of_trees)
