import pytest
from safeds.data.tabular import TaggedTable
from safeds.data.tabular import Table
from safeds.exceptions import LearningError
from safeds.ml.classification import GradientBoosting


def test_gradient_boosting_classification_fit() -> None:
    table = Table.from_csv("tests/resources/test_gradient_boosting_classification.csv")
    tagged_table = TaggedTable(table, "T")
    gradient_boosting_classification = GradientBoosting()
    gradient_boosting_classification.fit(tagged_table)
    assert True  # This asserts that the fit method succeeds


def test_gradient_boosting_classification_fit_invalid() -> None:
    table = Table.from_csv(
        "tests/resources/test_gradient_boosting_classification_invalid.csv"
    )
    tagged_table = TaggedTable(table, "T")
    gradient_boosting_classification = GradientBoosting()
    with pytest.raises(LearningError):
        gradient_boosting_classification.fit(tagged_table)
