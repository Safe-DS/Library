import pytest

from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import LearningError
from safeds.exceptions import PredictionError
from safeds.ml.classification import LogisticRegression
from tests.fixtures import resolve_resource_path


class TestFit:
    def test_logistic_regression_fit(self) -> None:
        table = Table.from_csv(resolve_resource_path("test_logistic_regression.csv"))
        tagged_table = TaggedTable(table, "T")
        log_regression = LogisticRegression()
        log_regression.fit(tagged_table)
        assert True  # This asserts that the fit method succeeds

    def test_logistic_regression_fit_invalid(self) -> None:
        table = Table.from_csv(
            resolve_resource_path("test_logistic_regression_invalid.csv")
        )
        tagged_table = TaggedTable(table, "T")
        log_regression = LogisticRegression()
        with pytest.raises(LearningError):
            log_regression.fit(tagged_table)


class TestPredict:
    def test_logistic_regression_predict(self) -> None:
        table = Table.from_csv(resolve_resource_path("test_logistic_regression.csv"))
        tagged_table = TaggedTable(table, "T")
        log_regression = LogisticRegression()
        log_regression.fit(tagged_table)
        log_regression.predict(tagged_table.features)
        assert True  # This asserts that the predict method succeeds

    def test_logistic_regression_predict_not_fitted(self) -> None:
        table = Table.from_csv(resolve_resource_path("test_logistic_regression.csv"))
        tagged_table = TaggedTable(table, "T")
        log_regression = LogisticRegression()
        with pytest.raises(PredictionError):
            log_regression.predict(tagged_table.features)

    def test_logistic_regression_predict_invalid(self) -> None:
        table = Table.from_csv(resolve_resource_path("test_logistic_regression.csv"))
        invalid_table = Table.from_csv(
            resolve_resource_path("test_logistic_regression_invalid.csv")
        )
        tagged_table = TaggedTable(table, "T")
        invalid_tagged_table = TaggedTable(invalid_table, "T")
        log_regression = LogisticRegression()
        log_regression.fit(tagged_table)
        with pytest.raises(PredictionError):
            log_regression.predict(invalid_tagged_table.features)
