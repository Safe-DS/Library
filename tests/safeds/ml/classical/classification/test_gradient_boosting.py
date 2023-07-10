import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import OutOfBoundsError
from safeds.ml.classical.classification import GradientBoosting


@pytest.fixture()
def training_set() -> TaggedTable:
    table = Table({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    return table.tag_columns(target_name="col1", feature_names=["col2"])


class TestNumberOfTrees:
    def test_should_be_passed_to_fitted_model(self, training_set: TaggedTable) -> None:
        fitted_model = GradientBoosting(number_of_trees=2).fit(training_set)
        assert fitted_model.number_of_trees == 2

    def test_should_be_passed_to_sklearn(self, training_set: TaggedTable) -> None:
        fitted_model = GradientBoosting(number_of_trees=2).fit(training_set)
        assert fitted_model._wrapped_classifier is not None
        assert fitted_model._wrapped_classifier.n_estimators == 2

    @pytest.mark.parametrize("number_of_trees", [-1, 0], ids=["minus_one", "zero"])
    def test_should_raise_if_less_than_1(self, number_of_trees: int) -> None:
        with pytest.raises(
            OutOfBoundsError,
            match=rf"number_of_trees \(={number_of_trees}\) is not inside \(0, \u221e\)\.",
        ):
            GradientBoosting(number_of_trees=number_of_trees)


class TestLearningRate:
    def test_should_be_passed_to_fitted_model(self, training_set: TaggedTable) -> None:
        fitted_model = GradientBoosting(learning_rate=2).fit(training_set)
        assert fitted_model.learning_rate == 2

    def test_should_be_passed_to_sklearn(self, training_set: TaggedTable) -> None:
        fitted_model = GradientBoosting(learning_rate=2).fit(training_set)
        assert fitted_model._wrapped_classifier is not None
        assert fitted_model._wrapped_classifier.learning_rate == 2

    @pytest.mark.parametrize("learning_rate", [-1, 0], ids=["minus_one", "zero"])
    def test_should_raise_if_less_than_or_equal_to_0(self, learning_rate: int) -> None:
        with pytest.raises(
            OutOfBoundsError,
            match=rf"learning_rate \(={learning_rate}\) is not inside \(0, \u221e\)\.",
        ):
            GradientBoosting(learning_rate=learning_rate)
