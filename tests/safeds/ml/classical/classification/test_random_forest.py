import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.ml.classical.classification import RandomForest


@pytest.fixture()
def training_set() -> TaggedTable:
    table = Table.from_dict({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    return table.tag_columns(target_name="col1", feature_names=["col2"])


class TestNumberOfTrees:
    def test_should_be_passed_to_fitted_model(self, training_set: TaggedTable) -> None:
        fitted_model = RandomForest(number_of_trees=2).fit(training_set)
        assert fitted_model._number_of_trees == 2

    def test_should_be_passed_to_sklearn(self, training_set: TaggedTable) -> None:
        fitted_model = RandomForest(number_of_trees=2).fit(training_set)
        assert fitted_model._wrapped_classifier is not None
        assert fitted_model._wrapped_classifier.n_estimators == 2

    def test_should_raise_if_less_than_or_equal_to_0(self) -> None:
        with pytest.raises(ValueError, match="The parameter 'number_of_trees' has to be greater than 0."):
            RandomForest(number_of_trees=-1)
