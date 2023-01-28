import pytest
from safeds.data import SupervisedDataset
from safeds.data.tabular import Table
from safeds.exceptions import PredictionError
from safeds.ml.classification import GradientBoosting


def test_gradient_boosting_predict() -> None:
    table = Table.from_csv("tests/resources/test_gradient_boosting_classification.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    gradient_boosting_classification = GradientBoosting()
    gradient_boosting_classification.fit(supervised_dataset)
    gradient_boosting_classification.predict(supervised_dataset.feature_vectors)
    assert True  # This asserts that the predict method succeeds


def test_gradient_boosting_predict_not_fitted() -> None:
    table = Table.from_csv("tests/resources/test_gradient_boosting_classification.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    gradient_boosting = GradientBoosting()
    with pytest.raises(PredictionError):
        gradient_boosting.predict(supervised_dataset.feature_vectors)


def test_gradient_boosting_predict_invalid() -> None:
    table = Table.from_csv("tests/resources/test_gradient_boosting_classification.csv")
    invalid_table = Table.from_csv(
        "tests/resources/test_gradient_boosting_classification_invalid.csv"
    )
    supervised_dataset = SupervisedDataset(table, "T")
    invalid_supervised_dataset = SupervisedDataset(invalid_table, "T")
    gradient_boosting = GradientBoosting()
    gradient_boosting.fit(supervised_dataset)
    with pytest.raises(PredictionError):
        gradient_boosting.predict(invalid_supervised_dataset.feature_vectors)
