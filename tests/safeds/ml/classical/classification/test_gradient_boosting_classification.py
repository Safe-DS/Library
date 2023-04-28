import pytest
from safeds.data.tabular.containers import Table
from safeds.ml.classical.classification import GradientBoosting


def test_should_throw_value_error_if_learning_rate_is_non_positive() -> None:
    with pytest.raises(ValueError, match="learning_rate must be positive."):
        GradientBoosting(learning_rate=-1)


def test_should_pass_learning_rate_to_sklearn() -> None:
    training_set = Table.from_dict({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    tagged_table = training_set.tag_columns("col1")

    regressor = GradientBoosting(learning_rate=2).fit(tagged_table)
    assert regressor._wrapped_classifier is not None
    assert regressor._wrapped_classifier.learning_rate == regressor._learning_rate
