import pytest
from safeds.data import SupervisedDataset
from safeds.data.tabular import Table
from safeds.exceptions import LearningError
from safeds.ml.regression import GradientBoosting


def test_gradient_boosting_regression_fit() -> None:
    table = Table.from_csv("tests/resources/test_gradient_boosting_regression.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    gradient_boosting = GradientBoosting()
    gradient_boosting.fit(supervised_dataset)
    assert True  # This asserts that the fit method succeeds


def test_gradient_boosting_regression_fit_invalid() -> None:
    table = Table.from_csv(
        "tests/resources/test_gradient_boosting_regression_invalid.csv"
    )
    supervised_dataset = SupervisedDataset(table, "T")
    gradient_boosting_regression = GradientBoosting()
    with pytest.raises(LearningError):
        gradient_boosting_regression.fit(supervised_dataset)
