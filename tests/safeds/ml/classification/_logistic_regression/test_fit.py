import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import LearningError
from safeds.ml.classification import LogisticRegression
from tests.fixtures import resolve_resource_path


def test_logistic_regression_fit() -> None:
    table = Table.from_csv(resolve_resource_path("test_logistic_regression.csv"))
    tagged_table = TaggedTable(table, "T")
    log_regression = LogisticRegression()
    log_regression.fit(tagged_table)
    assert True  # This asserts that the fit method succeeds


def test_logistic_regression_fit_invalid() -> None:
    table = Table.from_csv(
        resolve_resource_path("test_logistic_regression_invalid.csv")
    )
    tagged_table = TaggedTable(table, "T")
    log_regression = LogisticRegression()
    with pytest.raises(LearningError):
        log_regression.fit(tagged_table)
