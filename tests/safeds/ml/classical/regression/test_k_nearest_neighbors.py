import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.ml.classical.regression import KNearestNeighbors


@pytest.fixture()
def training_set() -> TaggedTable:
    table = Table.from_dict({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    return table.tag_columns(target_name="col1", feature_names=["col2"])


class TestNumberOfNeighbors:
    def test_should_be_passed_to_fitted_model(self, training_set: TaggedTable) -> None:
        fitted_model = KNearestNeighbors(number_of_neighbors=2).fit(training_set)
        assert fitted_model._number_of_neighbors == 2

    def test_should_be_passed_to_sklearn(self, training_set: TaggedTable) -> None:
        fitted_model = KNearestNeighbors(number_of_neighbors=2).fit(training_set)
        assert fitted_model._wrapped_regressor is not None
        assert fitted_model._wrapped_regressor.n_neighbors == 2

    def test_should_raise_if_less_than_or_equal_to_0(self) -> None:
        with pytest.raises(ValueError, match="The parameter 'number_of_neighbors' has to be greater than 0."):
            KNearestNeighbors(number_of_neighbors=-1)

    def test_should_raise_if_greater_than_sample_size(self, training_set: TaggedTable) -> None:
        with pytest.raises(ValueError, match="has to be less than or equal to the sample size"):
            KNearestNeighbors(number_of_neighbors=5).fit(training_set)
