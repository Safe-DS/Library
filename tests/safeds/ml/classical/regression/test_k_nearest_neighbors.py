import pytest
from safeds.data.labeled.containers import TaggedTable
from safeds.data.tabular.containers import Table
from safeds.exceptions import OutOfBoundsError
from safeds.ml.classical.regression import KNearestNeighborsRegressor


@pytest.fixture()
def training_set() -> TaggedTable:
    table = Table({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    return table.tag_columns(target_name="col1", feature_names=["col2"])


class TestNumberOfNeighbors:
    def test_should_be_passed_to_fitted_model(self, training_set: TaggedTable) -> None:
        fitted_model = KNearestNeighborsRegressor(number_of_neighbors=2).fit(training_set)
        assert fitted_model.number_of_neighbors == 2

    def test_should_be_passed_to_sklearn(self, training_set: TaggedTable) -> None:
        fitted_model = KNearestNeighborsRegressor(number_of_neighbors=2).fit(training_set)
        assert fitted_model._wrapped_regressor is not None
        assert fitted_model._wrapped_regressor.n_neighbors == 2

    @pytest.mark.parametrize("number_of_neighbors", [-1, 0], ids=["minus_one", "zero"])
    def test_should_raise_if_less_than_or_equal_to_0(self, number_of_neighbors: int) -> None:
        with pytest.raises(
            OutOfBoundsError,
            match=rf"number_of_neighbors \(={number_of_neighbors}\) is not inside \[1, \u221e\)\.",
        ):
            KNearestNeighborsRegressor(number_of_neighbors=number_of_neighbors)

    def test_should_raise_if_greater_than_sample_size(self, training_set: TaggedTable) -> None:
        with pytest.raises(ValueError, match="has to be less than or equal to the sample size"):
            KNearestNeighborsRegressor(number_of_neighbors=5).fit(training_set)
