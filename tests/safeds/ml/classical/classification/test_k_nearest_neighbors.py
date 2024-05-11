import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import OutOfBoundsError
from safeds.ml.classical.classification import KNearestNeighborsClassifier


@pytest.fixture()
def training_set() -> TabularDataset:
    table = Table({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    return table.to_tabular_dataset(target_name="col1")


class TestNumberOfNeighbors:
    def test_should_be_passed_to_fitted_model(self, training_set: TabularDataset) -> None:
        fitted_model = KNearestNeighborsClassifier(number_of_neighbors=2).fit(training_set)
        assert fitted_model.number_of_neighbors == 2

    def test_should_be_passed_to_sklearn(self, training_set: TabularDataset) -> None:
        fitted_model = KNearestNeighborsClassifier(number_of_neighbors=2).fit(training_set)
        assert fitted_model._wrapped_model is not None
        assert fitted_model._wrapped_model.n_neighbors == 2

    @pytest.mark.parametrize("number_of_neighbors", [-1, 0], ids=["minus_one", "zero"])
    def test_should_raise_if_less_than_or_equal_to_0(self, number_of_neighbors: int) -> None:
        with pytest.raises(
            OutOfBoundsError,
            match=rf"number_of_neighbors \(={number_of_neighbors}\) is not inside \[1, \u221e\)\.",
        ):
            KNearestNeighborsClassifier(number_of_neighbors=number_of_neighbors)

    def test_should_raise_if_greater_than_sample_size(self, training_set: TabularDataset) -> None:
        with pytest.raises(ValueError, match="has to be less than or equal to the sample size"):
            KNearestNeighborsClassifier(number_of_neighbors=5).fit(training_set)
