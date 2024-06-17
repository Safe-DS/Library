import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import OutOfBoundsError
from safeds.ml.classical.classification import KNearestNeighborsClassifier
from safeds.ml.hyperparameters import Choice


@pytest.fixture()
def training_set() -> TabularDataset:
    table = Table({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    return table.to_tabular_dataset(target_name="col1")


class TestNeighborCount:
    def test_should_be_passed_to_fitted_model(self, training_set: TabularDataset) -> None:
        fitted_model = KNearestNeighborsClassifier(neighbor_count=2).fit(training_set)
        assert fitted_model.neighbor_count == 2

    def test_should_be_passed_to_sklearn(self, training_set: TabularDataset) -> None:
        fitted_model = KNearestNeighborsClassifier(neighbor_count=2).fit(training_set)
        assert fitted_model._wrapped_model is not None
        assert fitted_model._wrapped_model.n_neighbors == 2

    @pytest.mark.parametrize("neighbor_count", [-1, 0, Choice(-1)], ids=["minus_one", "zero", "invalid_choice"])
    def test_should_raise_if_less_than_or_equal_to_0(self, neighbor_count: int | Choice[int]) -> None:
        with pytest.raises(OutOfBoundsError):
            KNearestNeighborsClassifier(neighbor_count=neighbor_count)

    def test_should_raise_if_greater_than_sample_size(self, training_set: TabularDataset) -> None:
        with pytest.raises(ValueError, match="has to be less than or equal to the sample size"):
            KNearestNeighborsClassifier(neighbor_count=5).fit(training_set)
