import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import PredictionError
from safeds.ml.classification import GradientBoosting
from tests.fixtures import resolve_resource_path


def test_gradient_boosting_predict() -> None:
    table = Table.from_csv(
        resolve_resource_path("test_gradient_boosting_classification.csv")
    )
    tagged_table = TaggedTable(table, "T")
    gradient_boosting_classification = GradientBoosting()
    gradient_boosting_classification.fit(tagged_table)
    gradient_boosting_classification.predict(tagged_table.features)
    assert True  # This asserts that the predict method succeeds


def test_gradient_boosting_predict_not_fitted() -> None:
    table = Table.from_csv(
        resolve_resource_path("test_gradient_boosting_classification.csv")
    )
    tagged_table = TaggedTable(table, "T")
    gradient_boosting = GradientBoosting()
    with pytest.raises(PredictionError):
        gradient_boosting.predict(tagged_table.features)


def test_gradient_boosting_predict_invalid() -> None:
    table = Table.from_csv(
        resolve_resource_path("test_gradient_boosting_classification.csv")
    )
    invalid_table = Table.from_csv(
        resolve_resource_path("test_gradient_boosting_classification_invalid.csv")
    )
    tagged_table = TaggedTable(table, "T")
    invalid_tagged_table = TaggedTable(invalid_table, "T")
    gradient_boosting = GradientBoosting()
    gradient_boosting.fit(tagged_table)
    with pytest.raises(PredictionError):
        gradient_boosting.predict(invalid_tagged_table.features)
