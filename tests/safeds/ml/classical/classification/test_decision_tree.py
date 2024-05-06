import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import OutOfBoundsError
from safeds.ml.classical.classification import DecisionTreeClassifier


@pytest.fixture()
def training_set() -> TabularDataset:
    table = Table({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    return table.to_tabular_dataset(target_name="col1")


class TestMaximumDepth:
    def test_should_be_passed_to_fitted_model(self, training_set: TabularDataset) -> None:
        fitted_model = DecisionTreeClassifier(maximum_depth=2).fit(training_set)
        assert fitted_model.maximum_depth == 2

    def test_should_be_passed_to_sklearn(self, training_set: TabularDataset) -> None:
        fitted_model = DecisionTreeClassifier(maximum_depth=2).fit(training_set)
        assert fitted_model._wrapped_classifier is not None
        assert fitted_model._wrapped_classifier.max_depth == 2

    @pytest.mark.parametrize("maximum_depth", [-1, 0], ids=["minus_one", "zero"])
    def test_should_raise_if_less_than_or_equal_to_0(self, maximum_depth: int) -> None:
        with pytest.raises(
            OutOfBoundsError,
            match=rf"maximum_depth \(={maximum_depth}\) is not inside \[1, \u221e\)\.",
        ):
            DecisionTreeClassifier(maximum_depth=maximum_depth)


class TestMinimumNumberOfSamplesInLeaves:
    def test_should_be_passed_to_fitted_model(self, training_set: TabularDataset) -> None:
        fitted_model = DecisionTreeClassifier(minimum_number_of_samples_in_leaves=2).fit(training_set)
        assert fitted_model.minimum_number_of_samples_in_leaves == 2

    def test_should_be_passed_to_sklearn(self, training_set: TabularDataset) -> None:
        fitted_model = DecisionTreeClassifier(minimum_number_of_samples_in_leaves=2).fit(training_set)
        assert fitted_model._wrapped_classifier is not None
        assert fitted_model._wrapped_classifier.min_samples_leaf == 2

    @pytest.mark.parametrize("minimum_number_of_samples_in_leaves", [-1, 0], ids=["minus_one", "zero"])
    def test_should_raise_if_less_than_or_equal_to_0(self, minimum_number_of_samples_in_leaves: int) -> None:
        with pytest.raises(
            OutOfBoundsError,
            match=rf"minimum_number_of_samples_in_leaves \(={minimum_number_of_samples_in_leaves}\) is not inside \[1, \u221e\)\.",
        ):
            DecisionTreeClassifier(minimum_number_of_samples_in_leaves=minimum_number_of_samples_in_leaves)
