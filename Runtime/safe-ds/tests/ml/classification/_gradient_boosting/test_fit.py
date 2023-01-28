import pytest
from safeds.data import SupervisedDataset
from safeds.data.tabular import Table
from safeds.exceptions import LearningError
from safeds.ml.classification import GradientBoosting


def test_gradient_boosting_classification_fit() -> None:
    table = Table.from_csv("tests/resources/test_gradient_boosting_classification.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    gradient_boosting_classification = GradientBoosting()
    gradient_boosting_classification.fit(supervised_dataset)
    assert True  # This asserts that the fit method succeeds


def test_gradient_boosting_classification_fit_invalid() -> None:
    table = Table.from_csv(
        "tests/resources/test_gradient_boosting_classification_invalid.csv"
    )
    supervised_dataset = SupervisedDataset(table, "T")
    gradient_boosting_classification = GradientBoosting()
    with pytest.raises(LearningError):
        gradient_boosting_classification.fit(supervised_dataset)
