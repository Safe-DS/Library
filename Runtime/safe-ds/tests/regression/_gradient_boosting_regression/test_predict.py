import pytest
from safeds.data import SupervisedDataset, Table
from safeds.exceptions import PredictionError
from safeds.regression import GradientBoosting


def test_gradient_boosting_predict() -> None:
    table = Table.from_csv("tests/resources/test_gradient_boosting_regression.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    gradient_boosting_regression = GradientBoosting()
    gradient_boosting_regression.fit(supervised_dataset)
    gradient_boosting_regression.predict(supervised_dataset.feature_vectors)
    assert True  # This asserts that the predict method succeeds


def test_gradient_boosting_predict_not_fitted() -> None:
    table = Table.from_csv("tests/resources/test_gradient_boosting_regression.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    gradient_boosting_regression = GradientBoosting()
    with pytest.raises(PredictionError):
        gradient_boosting_regression.predict(supervised_dataset.feature_vectors)


def test_gradient_boosting_predict_invalid() -> None:
    table = Table.from_csv("tests/resources/test_gradient_boosting_regression.csv")
    invalid_table = Table.from_csv(
        "tests/resources/test_gradient_boosting_regression_invalid.csv"
    )
    supervised_dataset = SupervisedDataset(table, "T")
    invalid_supervised_dataset = SupervisedDataset(invalid_table, "T")
    gradient_boosting_regression = GradientBoosting()
    gradient_boosting_regression.fit(supervised_dataset)
    with pytest.raises(PredictionError):
        gradient_boosting_regression.predict(invalid_supervised_dataset.feature_vectors)
