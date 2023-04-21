import pytest

from safeds.data.tabular.containers import Table
from safeds.ml.classical.regression import RandomForest


def test_number_of_trees_invalid() -> None:
    with pytest.raises(ValueError, match="The number of trees has to be greater than 0."):
        RandomForest(-1)


def test_number_of_trees_valid() -> None:
    training_set = Table.from_dict({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    tagged_training_set = training_set.tag_columns(target_name="col1", feature_names=["col2"])

    random_forest = RandomForest(10).fit(tagged_training_set)
    assert random_forest._wrapped_regressor is not None
    assert random_forest._wrapped_regressor.n_estimators == 10
