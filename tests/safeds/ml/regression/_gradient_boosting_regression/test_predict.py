import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import PredictionError
from safeds.ml.regression import GradientBoosting


def test_gradient_boosting_predict() -> None:
    table = Table.from_csv("tests/resources/test_gradient_boosting_regression.csv")
    tagged_table = TaggedTable(table, "T")
    gradient_boosting_regression = GradientBoosting()
    gradient_boosting_regression.fit(tagged_table)
    gradient_boosting_regression.predict(tagged_table.feature_vectors)
    assert True  # This asserts that the predict method succeeds


def test_gradient_boosting_predict_not_fitted() -> None:
    table = Table.from_csv("tests/resources/test_gradient_boosting_regression.csv")
    tagged_table = TaggedTable(table, "T")
    gradient_boosting_regression = GradientBoosting()
    with pytest.raises(PredictionError):
        gradient_boosting_regression.predict(tagged_table.feature_vectors)


def test_gradient_boosting_predict_invalid() -> None:
    table = Table.from_csv("tests/resources/test_gradient_boosting_regression.csv")
    invalid_table = Table.from_csv(
        "tests/resources/test_gradient_boosting_regression_invalid.csv"
    )
    tagged_table = TaggedTable(table, "T")
    invalid_tagged_table = TaggedTable(invalid_table, "T")
    gradient_boosting_regression = GradientBoosting()
    gradient_boosting_regression.fit(tagged_table)
    with pytest.raises(PredictionError):
        gradient_boosting_regression.predict(invalid_tagged_table.feature_vectors)
