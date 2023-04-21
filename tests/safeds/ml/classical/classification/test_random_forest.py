import pytest

from safeds.data.tabular.containers import Table
from safeds.ml.classical.classification import RandomForest


def test_number_of_trees_invalid() -> None:
    with pytest.raises(ValueError):
        RandomForest(-1)


def test_number_of_trees_valid() -> None:
    training_set = Table.from_dict({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    tagged_training_set = training_set.tag_columns(target_name="col1", feature_names=["col2"])

    random_forest = RandomForest(10).fit(tagged_training_set)
    assert (random_forest._wrapped_classifier.n_estimators == 10)
