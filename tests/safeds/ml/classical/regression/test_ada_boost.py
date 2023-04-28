import pytest
from safeds.data.tabular.containers import Table
from safeds.ml.classical.regression import AdaBoost


def test_should_throw_value_error() -> None:
    with pytest.raises(ValueError, match="learning_rate must be positive."):
        AdaBoost(learning_rate=-1)


def test_should_give_learning_rate_to_sklearn() -> None:
    training_set = Table.from_dict({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    tagged_table = training_set.tag_columns("col1")

    regressor = AdaBoost(learning_rate=2).fit(tagged_table)
    assert regressor._wrapped_regressor is not None
    assert regressor._wrapped_regressor.learning_rate == regressor.learning_rate
