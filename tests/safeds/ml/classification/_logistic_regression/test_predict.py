import pytest

from tests.fixtures import resolve_resource_path
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import PredictionError
from safeds.ml.classification import LogisticRegression


def test_logistic_regression_predict() -> None:
    table = Table.from_csv(resolve_resource_path("test_logistic_regression.csv"))
    tagged_table = TaggedTable(table, "T")
    log_regression = LogisticRegression()
    log_regression.fit(tagged_table)
    log_regression.predict(tagged_table.feature_vectors)
    assert True  # This asserts that the predict method succeeds


def test_logistic_regression_predict_not_fitted() -> None:
    table = Table.from_csv(resolve_resource_path("test_logistic_regression.csv"))
    tagged_table = TaggedTable(table, "T")
    log_regression = LogisticRegression()
    with pytest.raises(PredictionError):
        log_regression.predict(tagged_table.feature_vectors)


def test_logistic_regression_predict_invalid() -> None:
    table = Table.from_csv(resolve_resource_path("test_logistic_regression.csv"))
    invalid_table = Table.from_csv(
        resolve_resource_path("test_logistic_regression_invalid.csv")
    )
    tagged_table = TaggedTable(table, "T")
    invalid_tagged_table = TaggedTable(invalid_table, "T")
    log_regression = LogisticRegression()
    log_regression.fit(tagged_table)
    with pytest.raises(PredictionError):
        log_regression.predict(invalid_tagged_table.feature_vectors)
