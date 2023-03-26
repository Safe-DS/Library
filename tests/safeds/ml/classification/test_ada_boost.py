import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import LearningError
from safeds.ml.classification import AdaBoost
from safeds.exceptions import PredictionError
from tests.fixtures import resolve_resource_path


class TestFit:
    def test_ada_boost_fit(self) -> None:
        table = Table.from_csv(resolve_resource_path("test_ada_boost.csv"))
        tagged_table = TaggedTable(table, "T")
        ada_boost = AdaBoost()
        ada_boost.fit(tagged_table)
        assert True  # This asserts that the fit method succeeds

    def test_ada_boost_fit_invalid(self) -> None:
        table = Table.from_csv(resolve_resource_path("test_ada_boost_invalid.csv"))
        tagged_table = TaggedTable(table, "T")
        ada_boost = AdaBoost()
        with pytest.raises(LearningError):
            ada_boost.fit(tagged_table)


class TestPredict:
    def test_ada_boost_predict(self) -> None:
        table = Table.from_csv(resolve_resource_path("test_ada_boost.csv"))
        tagged_table = TaggedTable(table, "T")
        ada_boost = AdaBoost()
        ada_boost.fit(tagged_table)
        ada_boost.predict(tagged_table.features)
        assert True  # This asserts that the predict method succeeds

    def test_ada_boost_predict_not_fitted(self) -> None:
        table = Table.from_csv(resolve_resource_path("test_ada_boost.csv"))
        tagged_table = TaggedTable(table, "T")
        ada_boost = AdaBoost()
        with pytest.raises(PredictionError):
            ada_boost.predict(tagged_table.features)

    def test_ada_boost_predict_invalid(self) -> None:
        table = Table.from_csv(resolve_resource_path("test_ada_boost.csv"))
        invalid_table = Table.from_csv(resolve_resource_path("test_ada_boost_invalid.csv"))
        tagged_table = TaggedTable(table, "T")
        invalid_tagged_table = TaggedTable(invalid_table, "T")
        ada_boost = AdaBoost()
        ada_boost.fit(tagged_table)
        with pytest.raises(PredictionError):
            ada_boost.predict(invalid_tagged_table.features)
